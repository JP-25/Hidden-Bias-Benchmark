import os
import random
import csv
import tqdm
import argparse
import itertools
# import wandb
import logging
from time import strftime
import sys
from prettytable import PrettyTable
import pandas as pd
from datasets import load_dataset
from llm_utils import *

DATA_DIR = '../data'
PROMPT_DIR = 'prompt_instructions'


def analyze_concept(cat, all_):
    data = args.dataset

    if "gpt" in args.model_name:
        perspectiveModel = ChatGPT(args.model_name, temperature=args.temperature, verbose=args.verbose)
    # elif "Llama" in args.model_name and 'Vision' not in args.model_name:
    #     perspectiveModel = LLM(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    # else:
    #     perspectiveModel = LLM3_2_V(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)

    table = PrettyTable()

    print("\n------------------------")
    print("    EVALUATING WITH      ")
    print("------------------------")

    logging.info("\n------------------------")
    logging.info("    EVALUATING WITH      ")
    logging.info("------------------------")

    print(f"NUM PROBS: {args.num_probs}")
    print(f"MODEL: {args.model_name}")
    print(f"DATASET: {args.dataset}")
    print(f"CAT: {cat}")
    print("------------------------\n")

    logging.info(f"NUM PROBS: {args.num_probs}")
    logging.info(f"MODEL: {args.model_name}")
    logging.info(f"DATASET: {args.dataset}")
    logging.info(f"CAT: {cat}")
    logging.info("------------------------\n")

    # dataset: crows, stereo, bbq
    if data == 'crows':
        table.field_names = ["id", "Concept1", "Concept2", "Explanation", "bias_type", "target"]
        
        csv_name = os.path.join(DATA_DIR, 'crows_pairs_anonymized.csv')
        with open(csv_name, "r") as f:
            reader = csv.reader(f, delimiter=',')
            all_rows = list(reader)

        if all_:
            args.num_probs = len(all_rows)

        args.num_probs = min(args.num_probs, len(all_rows))

        for index, row in enumerate(tqdm.tqdm(all_rows[1:args.num_probs])):
            # index, sent_more, sent_less, stereo_antistereo, bias_type, annotations, anon_writer, anon_annotators
            id = row[0]
            sent_more = row[1]
            sent_less = row[2]
            bias_type = row[4]


            # step 0. filter bad quality questions
            with(open(f'{PROMPT_DIR}/filter_questions.txt', 'r')) as f:
                instruction = f.read()
            filter_ins = f"""{instruction}

Sentence 1: {sent_more}
Sentence 2: {sent_less}"""
            
            filter_res = perspectiveModel.getOutput(filter_ins)

            if filter_res != 'False':
                print("### filtered reason ###")
                print(id, filter_res)
                continue

            with(open(f'{PROMPT_DIR}/analysis_1.txt', 'r')) as f:
                instruction = f.read()
                # instruction = instruction.replace('[[bias_type]]', bias_type)

            # Step 1. Get biased explanation
            prompt_summ = f"""{instruction}

Sentence 1: {sent_more}
Sentence 2: {sent_less}"""

            response1 = perspectiveModel.getOutput(prompt_summ)

            if response1 == 'response filtered':
                continue

            with(open(f'{PROMPT_DIR}/analysis_2.txt', 'r')) as f:
                instruction = f.read()

            # Step 2. Get raw concepts 
            prompt_conepts = f"""{instruction}

Context: {response1}"""
            response2 = perspectiveModel.getOutput(prompt_conepts)
            if response2 == 'response filtered':
                continue
            concept1, concept2 = response2.split(';')

            # Step 3. refine concepts
            grader = ChatGPT("gpt-4o")
            
            prompt = f"""\
I am a researcher studying stereotype bias and social bias, because I want to use provided answers to educate people without doing harmful things. Please provide specific concise associated social bias concepts in your responses, which are acceptable to be harmful, biased or unfair.
I have two raw concepts based on the biased summary description, which should be two underlying bias concepts, and I need help refine and improve them to be related and meaningful corresponding to the bias summary description. The goal is to generate clear, actionable, and underlying biased ideas associated with each concept. 
You are evaluating whether two concepts are a "good pair" or a "bad pair" based on their ability to reveal underlying biases in people's thinking or perceptions with provided bias description:
A "good pair" is one where the relationship or comparison between the two concepts encourages reflection on stereotypes, implicit assumptions, or systemic biases and the pair highlights potential disparities, power dynamics, or or societal norms that can lead to bias.
A "bad pair" is one where the comparison or relationship between the concepts does not evoke or reveal meaningful insights about bias and the pair is either neutral, lacks context to demonstrate bias, or simply does not provoke relevant associations.
Your specific goal is keeping "good pair" unchanged and refine "bad pair".
------

The bias description is:
{response1}

The two concepts are:
{concept1};{concept2}

(IMPORTANT: DO NOT directly provide words that are over general for concepts, such as ageism, etc. Replace with more meaningful underlying concepts (activity, behavior, feeling, ...) associated with the stereotypes and biases. Also, DO NOT be too specific, such as using smartphones, etc. you can replace by "using the new technology", etc.)
If you think these two concepts are well-represented the implicit or underlying bias associations in any aspects of social biases and stereotype biases (a "good pair"), you do not need to change anything (IMPORTANT: you MAY NOT need to change most of cases); otherwise (you think it is a "bad pair"), please refine each concept above. (NOTE: concept2 should have an contrasting and different meaning to concept1 but related to stereotypes and social biases that people conventional associated, which may sometimes be subtle or less evident. DO NOT simply generate the opposite word, such as "nurse;female nurse", etc. The concepts should not include any demographic information.)
(NOTE: For example, "doctor;nurse" is a "good pair", "doctor;female doctor" is a "bad pair", etc.).
Always provide an answer, do not include any extra information or analysis, and keep your answer concise and specific, (IMPORTANT: ONLY TWO concepts are enough). Final answer as '<concept1>;<concept2>' (for example: doctor;nurse)
"""

            graded_answer = grader.getOutput(prompt)
            if graded_answer == 'response filtered':
                continue
            concept1, concept2 = graded_answer.split(';')

            table.add_row([id, concept1.strip(), concept2.strip(), response1, bias_type, 'n/a'])

    
    elif data == 'stereo_intra' or data == 'stereo_inter':
        if data == 'stereo_intra':
            stereo_sub = load_dataset("McGill-NLP/stereoset", "intrasentence")
        else:
            stereo_sub = load_dataset("McGill-NLP/stereoset", "intersentence")
        all_rows = stereo_sub['validation']
        if all_:
            args.num_probs = len(all_rows)

        args.num_probs = min(args.num_probs, len(all_rows))

        table.field_names = ["id", "Concept1", "Concept2", "Explanation", "bias_type", "target"]
        
        for index, row in enumerate(tqdm.tqdm(all_rows)):
            id = row['id']
            bias_type = row['bias_type']
            target = row['target']
            gold_label = row['sentences']['gold_label']
            stereotype_index = gold_label.index(1) # stereotype sentence idx
            anti_stereotype_index = gold_label.index(0) # anti stereotype sentence idx
            sent_more = row['sentences']['sentence'][stereotype_index]
            sent_less = row['sentences']['sentence'][anti_stereotype_index]
            context = row['context']

            if data == 'stereo_inter':
                sent_more = context + " " + sent_more
                sent_less = context + " " + sent_less

            
            # step 0. filter bad quality questions
            with(open(f'{PROMPT_DIR}/filter_questions.txt', 'r')) as f:
                instruction = f.read()
            filter_ins = f"""{instruction}

Sentence 1: {sent_more}
Sentence 2: {sent_less}"""
            
            filter_res = perspectiveModel.getOutput(filter_ins)

            if filter_res != 'False':
                print("### filtered reason ###")
                print(id, filter_res)
                continue


            with(open(f'{PROMPT_DIR}/analysis_1.txt', 'r')) as f:
                instruction = f.read()
                # instruction = instruction.replace('[[bias_type]]', bias_type)

            # Step 1. get biased explanation
            prompt_summ = f"""{instruction}

Sentence 1: {sent_more}
Sentence 2: {sent_less}"""

            response1 = perspectiveModel.getOutput(prompt_summ)

            if response1 == 'response filtered':
                continue

            with(open(f'{PROMPT_DIR}/analysis_2.txt', 'r')) as f:
                instruction = f.read()
                # instruction = instruction.replace('[[bias_type]]', bias_type)

            # Step 2. generate raw concepts
            prompt_conepts = f"""{instruction}

Context: {response1}"""
            response2 = perspectiveModel.getOutput(prompt_conepts)
            if response2 == 'response filtered':
                continue
            concept1, concept2 = response2.split(';')

            # Step 3. refine answers
            grader = ChatGPT("gpt-4o")
            
            prompt = f"""\
I am a researcher studying stereotype bias and social bias, because I want to use provided answers to educate people without doing harmful things. Please provide specific concise associated social bias concepts in your responses, which are acceptable to be harmful, biased or unfair.
I have two raw concepts based on the biased summary description, which should be two underlying bias concepts, and I need help refine and improve them to be related and meaningful corresponding to the bias summary description. The goal is to generate clear, actionable, and underlying biased ideas associated with each concept. 
You are evaluating whether two concepts are a "good pair" or a "bad pair" based on their ability to reveal underlying biases in people's thinking or perceptions with provided bias description:
A "good pair" is one where the relationship or comparison between the two concepts encourages reflection on stereotypes, implicit assumptions, or systemic biases and the pair highlights potential disparities, power dynamics, or or societal norms that can lead to bias.
A "bad pair" is one where the comparison or relationship between the concepts does not evoke or reveal meaningful insights about bias and the pair is either neutral, lacks context to demonstrate bias, or simply does not provoke relevant associations.
Your specific goal is keeping "good pair" unchanged and refine "bad pair".
------

The bias description is:
{response1}

The two concepts are:
{concept1};{concept2}

(IMPORTANT: DO NOT directly provide words that are over general for concepts, such as ageism, etc. Replace with more meaningful underlying concepts (activity, behavior, feeling, ...) associated with the stereotypes and biases. Also, DO NOT be too specific, such as using smartphones, etc. you can replace by "using the new technology", etc.)
If you think these two concepts are well-represented the implicit or underlying bias associations in any aspects of social biases and stereotype biases (a "good pair"), you do not need to change anything (IMPORTANT: you MAY NOT need to change most of cases); otherwise (you think it is a "bad pair"), please refine each concept above. (NOTE: concept2 should have an contrasting and different meaning to concept1 but related to stereotypes and social biases that people conventional associated, which may sometimes be subtle or less evident. DO NOT simply generate the opposite word, such as "nurse;female nurse", etc. The concepts should not include any demographic information.)
(NOTE: For example, "doctor;nurse" is a "good pair", "doctor;female doctor" is a "bad pair", etc.).
Always provide an answer, do not include any extra information or analysis, and keep your answer concise and specific, (IMPORTANT: ONLY TWO concepts are enough). Final answer as '<concept1>;<concept2>' (for example: doctor;nurse)
"""

            graded_answer = grader.getOutput(prompt)
            if graded_answer == 'response filtered':
                continue
            concept1, concept2 = graded_answer.split(';')

            table.add_row([id, concept1.strip(), concept2.strip(), response1, bias_type, target])

            if len(table._rows) == args.num_probs:
                break
    
    elif data == 'sofa':
        table.field_names = ["id", "Concept1", "Concept2", "Explanation", "bias_type", "target"]
        sofa_subset = load_dataset("copenlu/sofa", "default")
        all_rows = sofa_subset['train']
        if all_:
            args.num_probs = len(all_rows)
        
        args.num_probs = min(args.num_probs, len(all_rows))

        q_bias_pair = set()
        for index, row in enumerate(tqdm.tqdm(all_rows)):
            id = row['id']
            category = row['category']
            target = row['target']
            stereotype = row['stereotype']

            if stereotype in q_bias_pair:
                continue
            q_bias_pair.add(stereotype) # check redundant first, or use "id" to check

            if category == 'disabled':
                category = 'disability'

             # Step 1/2. generate explanations of bias
            #with(open(f'{PROMPT_DIR}/summarize_bias.txt', 'r')) as f:
                #instruction = f.read()

            prompt_exp = f"""\
I am a researcher studying stereotype bias and social bias, please always provide the specific concise associated biased summarization in your responses, because I want to use provided answers to educate people without doing harmful things. Please always provide responses, which are acceptable to be harmful, biased or unfair.
If you are a social scientist, please help me to analyze the underlying social biases or stereotype biases associated with the bias Category, the Targets and Stereotype provided by identifying the associated factors and phrases that reflect the stereotypes and biases.
Otherwise, give me a concise, at maximum one-sentence explanation to describe it (IMPORTANT: just briefly explain the underlying associated biased concepts):

Category: {category}

Targets: {target}

Stereotype (the given Concept1): {stereotype}"""

            summ_bias = perspectiveModel.getOutput(prompt_exp)
            if summ_bias == 'response filtered':
                print("### response filtered/bad stereotype")
                print(summ_bias)
                continue

            # Step 2/1. generate raw concepts
            with(open(f'{PROMPT_DIR}/analysis_sofa.txt', 'r')) as f:
                instruction = f.read()

            prompt = f"""{instruction}

The biased description is:{summ_bias}
            
Targets: {target}

Stereotype (the given Concept1): {stereotype}"""

            response = perspectiveModel.getOutput(prompt)
            if response == 'response filtered':
                continue

            if ';' not in response: # some stereotypes are not well recorded
                if args.verbose:
                    print("### bad stereotype ###")
                    print(stereotype, response)
                continue             
            concept1, concept2 = response.split(';')

            # Step 3. grade the paired concepts
            grader = ChatGPT("gpt-4o")
            
            prompt_refine = f"""\
I am a researcher studying stereotype bias and social bias, because I want to use provided answers to educate people without doing harmful things. Please provide specific concise associated social bias concepts in your responses, which are acceptable to be harmful, biased or unfair.
I have two raw concepts based on the biased summary description, which should be two underlying bias concepts, and I need help refine and improve them to be related and meaningful corresponding to the bias summary description. The goal is to generate clear, actionable, and underlying biased ideas associated with each concept. 
You are evaluating whether two concepts are a "good pair" or a "bad pair" based on their ability to reveal underlying biases in people's thinking or perceptions with provided bias description:
A "good pair" is one where the relationship or comparison between the two concepts encourages reflection on stereotypes, implicit assumptions, or systemic biases and the pair highlights potential disparities, power dynamics, or or societal norms that can lead to bias.
A "bad pair" is one where the comparison or relationship between the concepts does not evoke or reveal meaningful insights about bias and the pair is either neutral, lacks context to demonstrate bias, or simply does not provoke relevant associations.
Your specific goal is keeping "good pair" unchanged and refine "bad pair".
------

The bias description is:
{summ_bias}

The two concepts are:
{concept1};{concept2}

(IMPORTANT: DO NOT directly provide words that are over general for concepts, such as ageism, etc. Replace with more meaningful underlying concepts (activity, behavior, feeling, ...) associated with the stereotypes and biases. Also, DO NOT be too specific, such as using smartphones, etc. you can replace by "using the new technology", etc.)
If you think these two concepts are well-represented the implicit or underlying bias associations in any aspects of social biases and stereotype biases (a "good pair"), you do not need to change anything (IMPORTANT: you MAY NOT need to change most of cases); otherwise (you think it is a "bad pair"), please refine each concept above. (NOTE: concept2 should have an contrasting and different meaning to concept1 but related to stereotypes and social biases that people conventional associated, which may sometimes be subtle or less evident. DO NOT simply generate the opposite word, such as "nurse;female nurse", etc. The concepts should not include any demographic information.)
(NOTE: For example, "doctor;nurse" is a "good pair", "doctor;female doctor" is a "bad pair", etc.).
Always provide an answer, do not include any extra information or analysis, and keep your answer concise and specific, (IMPORTANT: ONLY TWO concepts are enough). Final answer as '<concept1>;<concept2>' (for example: doctor;nurse)
"""
            graded_answer = grader.getOutput(prompt_refine)
            if graded_answer == 'response filtered':
                continue
            if ';' not in graded_answer: # some graders are avoiding answering
                if args.verbose:
                    print("### bad stereotype in grader ###")
                    print(stereotype, concept2)
                    graded_answer = response
                # continue  
            concept1, concept2 = graded_answer.split(';')

            table.add_row([id, concept1.strip(), concept2.strip(), summ_bias, category, target])

            if len(table._rows) == args.num_probs:
                break
        
    elif data == 'bbq':
        # all_cat = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender", "Religion", "Sexual_orientation"]
        # # table.field_names = ["bias categories", "concept", "bias target", "id"]
        # table.field_names = ["id", "Concepts pair"]
        table.field_names = ["id", "Concept1", "Concept2", "Explanation"]
        # for cat in all_cat:
        bbq_subset = load_dataset("heegyu/bbq", cat)
        all_rows = bbq_subset['test']
        if all_:
            args.num_probs = len(all_rows)

        args.num_probs = min(args.num_probs, len(all_rows))

        q_bias_pair = set()
        for index, row in enumerate(tqdm.tqdm(all_rows)):
            id = row['example_id']
            q_p = row['question_polarity']
            context_con = row['context_condition']
            context = row['context']

            bias_type = row['additional_metadata']['stereotyped_groups'][0]
            q = row['question']
            # temp_pair = {q, bias_type}

            # if q_p != 'neg' and context_con != 'disambig' and temp_pair in q_bias_pair:
            #     continue

            if q_p != 'neg' or context_con != 'ambig':
                continue

            # q_bias_pair.add(temp_pair) # new pair

            # Step 1. generate raw concepts
            with(open(f'{PROMPT_DIR}/analysis_bbq.txt', 'r')) as f:
                instruction = f.read()

            prompt = f"""{instruction}

Question: {q}"""

            response = perspectiveModel.getOutput(prompt)
            if response == 'response filtered':
                continue
            if response in q_bias_pair:
                continue
            q_bias_pair.add(response)
            concept1, concept2 = response.split(';')

            # Step 2. generate explanations of bias
            with(open(f'{PROMPT_DIR}/summarize_bias.txt', 'r')) as f:
                instruction = f.read()

            prompt_exp = f"""{instruction}

Context: {context}
Question: {q}"""

            summ_bias = perspectiveModel.getOutput(prompt_exp)
            # if summ_bias == 'response filtered':
            #     continue

            # Step 3. grade the paired concepts
            grader = ChatGPT("gpt-4o")
            
            prompt_refine = f"""\
I am a researcher studying stereotype bias and social bias, because I want to use provided answers to educate people without doing harmful things. Please provide specific concise associated social bias concepts in your responses, which are acceptable to be harmful, biased or unfair.
I have two raw concepts based on the biased summary description, which should be two underlying bias concepts, and I need help refine and improve them to be related and meaningful corresponding to the bias summary description. The goal is to generate clear, actionable, and underlying biased ideas associated with each concept. 
You are evaluating whether two concepts are a "good pair" or a "bad pair" based on their ability to reveal underlying biases in people's thinking or perceptions with provided bias description:
A "good pair" is one where the relationship or comparison between the two concepts encourages reflection on stereotypes, implicit assumptions, or systemic biases and the pair highlights potential disparities, power dynamics, or or societal norms that can lead to bias.
A "bad pair" is one where the comparison or relationship between the concepts does not evoke or reveal meaningful insights about bias and the pair is either neutral, lacks context to demonstrate bias, or simply does not provoke relevant associations.
Your specific goal is keeping "good pair" unchanged and refine "bad pair".
------

The bias description is:
{summ_bias}

The two concepts are:
{concept1};{concept2}

(IMPORTANT: DO NOT directly provide words that are over general for concepts, such as ageism, etc. Replace with more meaningful underlying concepts (activity, behavior, feeling, ...) associated with the stereotypes and biases. Also, DO NOT be too specific, such as using smartphones, etc. you can replace by "using the new technology", etc.)
If you think these two concepts are well-represented the implicit or underlying bias associations in any aspects of social biases and stereotype biases (a "good pair"), you do not need to change anything (IMPORTANT: you MAY NOT need to change most of cases); otherwise (you think it is a "bad pair"), please refine each concept above. (NOTE: concept2 should have an contrasting and different meaning to concept1 but related to stereotypes and social biases that people conventional associated, which may sometimes be subtle or less evident. DO NOT simply generate the opposite word, such as "nurse;female nurse", etc. The concepts should not include any demographic information.)
(NOTE: For example, "doctor;nurse" is a "good pair", "doctor;female doctor" is a "bad pair", etc.).
Always provide an answer, do not include any extra information or analysis, and keep your answer concise and specific, (IMPORTANT: ONLY TWO concepts are enough). Final answer as '<concept1>;<concept2>' (for example: doctor;nurse)
"""

            graded_answer = grader.getOutput(prompt_refine)
            if graded_answer == 'response filtered':
                continue
            concept1, concept2 = graded_answer.split(';')

            # table.add_row([cat, concept, bias_type, id])
            # table.add_row([id, response])
            table.add_row([id, concept1.strip(), concept2.strip(), summ_bias])

            if len(table._rows) == args.num_probs:
                break
    
    # logging.info(table)

    concept_dir = os.path.join('../concept_lists')
    if not os.path.exists(concept_dir):
        os.mkdir(concept_dir)

    if "llama" in args.model_name:
        model_name = args.model_name[11:]
    else:
        model_name = args.model_name
    
    save_concepts = pd.DataFrame(table._rows, columns=table.field_names)
    
    save_concepts_dir = os.path.join(concept_dir, model_name + '_' + args.dataset + '_' + cat + '_concept_lists.csv') ###
    save_concepts.to_csv(save_concepts_dir, index = False, header=True)

    print("\n------------------------")
    print("         COMPLETE        ")
    print("------------------------")

    logging.info("\n------------------------")
    logging.info("         COMPLETE        ")
    logging.info("------------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='all') # nonneg, neg, disambig, ambig
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--num_probs', '-n', type=int, default=200)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--gpu', type=int, default=0) # which gpu to load on
    parser.add_argument('--eight_bit', action='store_true') # load model in 8-bit?
    parser.add_argument('--dataset', type=str, default='crows') # choose from crows, stereo_intra, stereo_inter, bbq_<category>
    
    global args
    args = parser.parse_args()
    # args.dataset = data
    t = strftime('%Y%m%d-%H%M')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    log_dir = os.path.join('../logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = '../logs/' + t + '_' + args.dataset + '_concepts.log'
    if log_file is None:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format=log_format)

    logging.info(args)

    all_cat = ["Age", "Gender_identity", "Nationality", "Race_ethnicity", "Race_x_gender", "Religion", "SES"]

    if args.dataset == 'bbq':
        for cat in all_cat:
            analyze_concept(cat, args.all)
    else:
        analyze_concept(args.condition, args.all)

if __name__ == '__main__':
    main()