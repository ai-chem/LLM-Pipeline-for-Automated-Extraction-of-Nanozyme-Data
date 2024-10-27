import glob
import json
import os
from pathlib import Path
from time import time

import click
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, tool
from langchain_openai import ChatOpenAI
from data_preproccessing.pdf2txt import extract_text_from_pdf
from tqdm import tqdm

from logger import LOGGER
from utils import merge_dictionaries

load_dotenv(override=True)

LOGGER.info(f'OPENAI_API_KEY = "{os.getenv("OPENAI_API_KEY")}"')


def get_files_with_extension(directory, extension):
    return glob.glob(os.path.join(directory, f"*{extension}"))


@click.command()
@click.argument("pdf_articles_dir", type=click.Path())
@click.argument("pdf_supplements_dir", type=click.Path())
@click.argument("ner_json_dir", type=click.Path())
@click.argument("results_dir", type=click.Path())
def main(
    pdf_articles_dir: str, pdf_supplements_dir: str, ner_json_dir: str, results_dir: str
):
    directory = str(Path(pdf_articles_dir))

    extension = ".pdf"
    articles_files = get_files_with_extension(directory, extension)
    LOGGER.info(f"Files count: {len(articles_files)}")

    for article_file in tqdm(articles_files):
        try:
            start_time = time()
            LOGGER.info(
                f"Agent initialization start: {article_file[len(directory)+1:]}"
            )

            text_dict = {}

            text_dict["article_text"] = extract_text_from_pdf(article_file)

            supplement_file = None
            si_file_path = f"{pdf_supplements_dir}/{article_file[len(directory)+1:]}"
            if os.path.isfile(si_file_path):
                supplement_file = si_file_path
            text_dict["supplement_text"] = (
                extract_text_from_pdf(supplement_file) if supplement_file else None
            )

            @tool("get_full_text")
            def get_full_text(query: str) -> str:
                "Returns full text of the article and supplement information if provided."

                full_text_dict = {}
                full_text_dict["article_text"] = text_dict["article_text"]
                full_text_dict["supplement_text"] = text_dict["supplement_text"]

                return "```json\n" + str(full_text_dict) + "\n```"

            @tool("llm_extractor")
            def llm_extractor(file_name: str) -> str:
                "Extracts Named Entities from article with the help of mistral-7b and llama3-8b. Can be useful when you need to get parameters of the experiments. WARNING: this tool does not separate parameters by experiment and may miss some important parameters. You should use this tool first, and then be sure to check everything and fix it with the full text."

                article_name = article_file[len(directory) + 1 : -4]
                mistral_article_dict = {}
                mistral_si_dict = {}

                try:
                    with open(
                        f"{ner_json_dir}/{article_name}_mistral.json",
                        "r",
                        encoding="utf-8",
                    ) as file:
                        mistral_article_dict = json.load(file)
                except FileNotFoundError:
                    LOGGER.warning(f"Can't find file {article_name}_mistral.json")
                    mistral_article_dict = {}

                try:
                    with open(
                        f"{ner_json_dir}/{article_name}_si_mistral.json",
                        "r",
                        encoding="utf-8",
                    ) as file:
                        mistral_si_dict = json.load(file)
                except FileNotFoundError:
                    LOGGER.warning(f"Can't find file {article_name}_si_mistral.json")
                    mistral_si_dict = {}

                mistral_dict = merge_dictionaries(
                    [mistral_article_dict, mistral_si_dict]
                )

                llama_article_dict = {}
                llama_si_dict = {}

                try:
                    with open(
                        f"{ner_json_dir}/{article_name}_lama.json",
                        "r",
                        encoding="utf-8",
                    ) as file:
                        llama_article_dict = json.load(file)
                except FileNotFoundError:
                    LOGGER.warning(f"Can't find file {article_name}_lama.json")
                    llama_article_dict = {}

                try:
                    with open(
                        f"{ner_json_dir}/{article_name}_si_lama.json",
                        "r",
                        encoding="utf-8",
                    ) as file:
                        llama_si_dict = json.load(file)
                except FileNotFoundError:
                    LOGGER.warning(f"Can't find file {article_name}_si_lama.json")
                    llama_si_dict = {}

                llama_dict = merge_dictionaries([llama_article_dict, llama_si_dict])

                return (
                    "```"
                    + "\nmistral-7b answer:\n"
                    + str(mistral_dict)
                    + "\nllama3-8b answer:\n"
                    + str(llama_dict)
                    + "\n```"
                )

            tools = [get_full_text, llm_extractor]

            agent_llm = ChatOpenAI(
                temperature=0,
                model="gpt-4o",
                streaming=True,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            prompt = """
            You are helpful assistant in chemistry, specializing in nanozymes. Your task is to analyze scientific articles and extract detailed information about various experiments with nanozymes. It is crucial for you to accurately and comprehensively describe each experiment separately, without referring to other experiments in the article.
            Usually, the articles contain several experiments with nanozymes with different parameters, such as formula, activity (usually peroxidase, oxidase, catalase or laccase), syngony (usually cubic, hexagonal, tetragonal, monoclinic, orthorhombic, trigonal, amorphous or triclinic), length, width and depth (or just size or diameter), surface chemistry (naked by default or poly(ethylene oxide), poly(N-Vinylpyrrolidone), Tetrakis(4-carboxyphenyl)porphine or other), polymer used in synthesis (none or poly(N-Vinylpyrrolidone), oleic acid, poly(ethylene oxide), BSA or other), surfactant (none or l-ascorbic acid, ethylene glycol, sodium citrate, cetrimonium bromide, citric acid, trisodium citrate, ascorbic acid or other), molar mass, Michaelis constant Km, molar maximum reaction rate Vmax, reaction type (substrat + co-substrat) (TMB + H2O2, H2O2 + TMB, TMB, ABTS + H2O2, H2O2, OPD + H2O2, H2O2 + GSH or other), minimum concentration of the substrate when measuring catalytic activity C_min (mM), maximum concentration of the substrate when measuring catalytic activity C_max (mM), concentration of the co-substrate when measuring the catalytic activity (mM), concentration of nanoparticles in the measurement of catalytic (mg/mL), pH at which the catalytic activity was measured and temperature at which the research was carried out (°C). You need to find all the experiments with different values mentioned in the article and write about each of them separately. It's imperative that each of these elements is addressed independently for every experiment, providing a complete and isolated description with accurate measurements in appropriate units. This approach will ensure a comprehensive and clear understanding of each experiment as an individual entity within the scientific literature on nanozymes.
            You should describe the experiments in the article separately in words, while keeping the numerical values in the right units of measurement. It is critically important to extract all the numerical values as in the example, especially important are formula, activity, syngony, length, width, depth (size or diameter), Km, Vmax, reaction type. Usually such parameters as Michaelis constant  Km (mM), Vmax, mM/s are obtained in two experiments for every type of nanoparticle. You must determine what type of reaction such parameters as Michaelis constant Km (mM), Vmax, mM/s belong to. Reaction type is H2O2+TMB when H2O2 is a substrate and TMB in co-cubstrate. Reaction type is TMB +H2O2 when  TMB  is a substrate and H2O2 in co-cubstrate.For example in pair H2O2 and TMB in first case (you call this case as Reaction type TMB+H2O2) H2O2  plays role as a co-substrate with  constant concentration(C(const), mM) and TMB   plays role as a substrate  with concentrations from  Cmin,mM to  Cmax, mM. In second case (you call this case as Reaction type H2O2+)TMB) TMB  plays role as a co-substrate with  constant concentration(C(const), mM) and H2O2    plays role as a substrate  with concentrations from  Cmin,mM to  Cmax, mM. Please divide all the data into 2 tracks: where H2O2 was a substrate and its concentration varied and where H2O2 was a co-substrate and had a constant concentration. Please show data only for those nanoparticles for which the kinetic assay was performed. All other parameters from the example are also important.
            To answer questions, you can use the following tools: get_full_text, llm_extractor. If you have not managed to obtain the information you needed to answer with the help of the tools, mention that in your final answer. Do not attempt to use knowledge you already have. Don't use tools if you don't need them. After you receive a response from the tool, be sure to write your thoughts. Answer in the language that the question is asked in. Before writing the final answer, don't forget to write 'Final answer:'. You need to write this before you write the text of the final answer.
            You must write complete answer after phrase 'Final Answer:'"""
            agent = initialize_agent(
                tools,
                agent_llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                agent_kwargs={"prefix": prompt, "seed": 42},
            )

            LOGGER.info(
                f"Successful Agent initialization: {article_file[len(directory)+1:]}"
            )

            user_prompt = (
                "get all parameters, use only get_full_text and llm_extractor tools"
            )
            for i in range(5):
                try:
                    response = agent.run(user_prompt)
                    with open(
                        f"{results_dir}/{article_file[len(directory)+1:-4]}.md",
                        "w",
                    ) as f:
                        f.write(response)
                    break
                except Exception as e:
                    LOGGER.error(e)
                    with open(
                        f"{results_dir}/{article_file[len(directory)+1:-4]}.md",
                        "w",
                    ) as f:
                        f.write(f"Error: \n{str(e)}")

            end_time = time()

            LOGGER.info(
                f"Article {article_file[len(directory)+1:]} was processed in {end_time-start_time:.2f}s"
            )
            LOGGER.info("")
        except Exception as e:
            LOGGER.error(e)
            with open(
                f"{results_dir}/{article_file[len(directory)+1:-4]}.md", "w"
            ) as f:
                f.write(f"Error: \n{str(e)}")


if __name__ == "__main__":
    main()
