
""" TEST pour ce fichier """

from __future__ import absolute_import, division, print_function

import json
import logging
import os
import pandas as pd

import nltk
nltk.download('punkt')


QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]

def _get_correct_alignement(context, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       # When the gold label position is good
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1   # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2   # When the gold label is off by two character
    elif context.find(gold_text) != -1:
        return context.find(gold_text), context.find(gold_text) + len(gold_text)
    else:
        print(answer, "    ", context)
        raise ValueError()

def process_qa_text(context, question, answer):
    ans_gen_input = f"question: {question}  context: {context}"
    ans_gen_target = f"{answer}"
    return {"source_text": ans_gen_input, "target_text": ans_gen_target, "task": "qa"}

def process_qg_text(qg_format, context, question, answer):
    answer_text = answer['text'].strip()

    if qg_format == "prepend":
        que_gen_input = f"answer: {answer_text}  context: {context}"
    elif qg_format == "highlight":
        start_pos, end_pos = _get_correct_alignement(context, answer)
        que_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
    else:
        start_pos, end_pos = _get_correct_alignement(context, answer)
        que_gen_input = f"answer: {answer_text} context: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"

    que_gen_target = f"{question}"
    return {"source_text": que_gen_input, "target_text": que_gen_target, "task": "qg"}

def process_e2e_qg(paragraph):
    source_text = f"generate questions: {paragraph['context'].strip()}"
    questions = [qas['question'].strip() for qas in paragraph['qas']]
    target_text = " {sep_token} ".join(questions)
    target_text = f"{target_text} {{sep_token}}"
    return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}

def process_ans_ext(paragraph):
    context = paragraph['context'].strip()

    # split into sentences
    sents = nltk.sent_tokenize(context)

    # get positions of the sentences
    positions = []
    for i, sent in enumerate(sents):
        if i == 0:
            start, end = 0, len(sent)
        else:
            start, end = (prev_end + 1), (prev_end + len(sent) + 1)
        prev_end = end
        positions.append({'start': start, 'end': end})

    # get answers
    answers = [qa['answers'][0] for qa in paragraph['qas']]

    # get list of answers for each sentence
    sent_answers = []
    for pos, sent in zip(positions, sents):
        target_answers = []
        for ans in answers:
            if ans['answer_start'] in range(pos['start'], pos['end']):
                target_answers.append(ans['text'].strip())
        sent_answers.append(target_answers)

    # build inputs and targets
    examples = []
    for i, ans in enumerate(sent_answers):
        context = "extract answers:"
        if len(ans) == 0 : continue
        ans = list(set(ans))
        for j, sent in enumerate(sents):
            if i == j:
                sent = "{hl_token} %s {hl_token}" % sent
            context = "%s %s" % (context, sent)
            context = context.strip()
        input_text = context
        target_text = " {sep_token} ".join(ans) + " {sep_token}"

        examples.append({'source_text': input_text, "target_text": target_text, "task": "ans_ext"})

    return examples

def _generate_examples(filepath, qg_format):
    """This function returns the examples in the raw (text) form."""
    print("yo")
    print(filepath)
    logging.info("generating examples from = %s", filepath)
    count = 0
    tasks = ['qa', 'qg', 'ans_ext', 'e2e_qg']

    with open(filepath, encoding="utf-8") as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "").strip()
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()

                if 'ans_ext' in tasks:
                    ans_ext_examples = process_ans_ext(paragraph)
                    for example in ans_ext_examples:
                            yield count, example
                            count += 1

                if 'e2e_qg' in tasks:
                    yield count, process_e2e_qg(paragraph)
                    count += 1

                for qa in paragraph["qas"]:
                    question = qa["question"].strip()
                    id_ = qa["id"]

                    answers = [answer["text"].strip() for answer in qa["answers"]]
                    for task in tasks:
                        if task == 'qa':
                            yield count, process_qa_text(context, question, answers[0])
                            count += 1

                        if task == 'qg':
                            yield count, process_qg_text(qg_format, context, question, qa["answers"][0])
                            count += 1



cwd = os.getcwd()
print(cwd)


""" supprimer les lignes avec answers null"""





fquad_generator = _generate_examples("C:/Users/bapti/PycharmProjects/Picarre/PI2/question_generation-master/data/squad_multitask/test.json", "highlight")
print("yes")
print(type(fquad_generator))
sources, targets, tasks = [], [], []

elem = next(fquad_generator, None)

while elem is not None:
    sources.append(elem[1]['source_text'])
    targets.append(elem[1]['target_text'])
    tasks.append(elem[1]['task'])
    elem = next(fquad_generator, None)
df_train_fquad = pd.DataFrame({"source_text": sources, 'target_text': targets, "task": tasks})

df_train_fquad.to_csv("fquad_train.csv")


"""

fquad_generator = _generate_examples(os.path.join(args.path_fquad_data, "valid.json"), "highlight")
sources, targets, tasks = [], [], []
elem = next(fquad_generator, None)
while elem is not None:
    sources.append(elem[1]['source_text'])
    targets.append(elem[1]['target_text'])
    tasks.append(elem[1]['task'])
    elem = next(fquad_generator, None)
df_valid = pd.DataFrame({"source_text": sources, 'target_text': targets, "task": tasks})
df_valid.to_csv("fquad_valid.csv")


"""