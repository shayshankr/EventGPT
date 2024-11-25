from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
from tqdm import tqdm
import os
import json


prompt = """Role and Objective:
You are an intelligent evaluator tasked with assessing the understanding of image-based descriptions in generated responses for image-related question-answer pairs.

Your Task:
Compare the Predicted Answer with the Correct Answer to determine if the generated response aligns with the main elements, themes, and nuances of the image.

Instructions:

	•	Assess Relevance: Check if the Predicted Answer captures the primary elements, general theme, and main ideas present in the image.
	•	Prioritize Key Details: Focus on key details that define the scene, while minor or additional details are less critical.
	•	Consider Paraphrasing and Flexibility: Accept synonyms, paraphrases, or equivalent expressions as long as they reasonably match the main ideas.
	•	Assess Understanding Level: Gauge the Predicted Answer’s accuracy in reflecting an understanding of the image without requiring exact word-for-word matches.

Evaluation Criteria (Scoring Guide):

Score the Predicted Answer based on alignment with the Correct Answer using the following scale:

	•	5.0: Perfectly aligns, capturing all essential elements and themes with clarity and accuracy.
	•	4.0-4.9: Strong alignment, capturing most elements and themes, with minor deviations.
	•	3.0-3.9: Good alignment, capturing main elements and themes but missing minor details.
	•	2.0-2.9: Moderate alignment, capturing a few important aspects, missing some significant themes.
	•	1.0-1.9: Limited alignment, capturing very few aspects, with major parts misunderstood or missing.
	•	0.0-0.9: Minimal to no alignment, largely irrelevant or incorrect in terms of the image’s content.

Provide Evaluation in the Following Format:

Evaluate the following image-based question-answer pair:

	•	Question: {question}
	•	Correct Answer: {answer}
	•	Predicted Answer: {pred}

Your response should be a Python dictionary in the form of a string with the key 'score' and a floating-point value between 0.0 and 5.0, where 5.0 indicates the highest level of understanding.

Example Output Format:
{{'score': 4.3}}

Only provide the dictionary response in this format, without any additional explanation."""

result_template = {
    "query": "",
    "pred": "",
    "answer": "",
    "score": 0
}
def main():
    pipe = pipeline("/data1/SyL/model/Qwen2-72B-Instruct",
                    backend_config=TurbomindEngineConfig(tp=4, session_len=4096, dtype='bfloat16'))
    gen_config = GenerationConfig(temperature=0, max_new_tokens=2048)

    pred_json_file_path = "/data/SyL/EventChat/ablation/no-st/pred/event_caption_result.json"
    result_caption_question_file_path = "/data/SyL/EventChat/ablation/no-st/result/event_caption_eval.json"
    with open(pred_json_file_path, 'r') as f:
        pred_data = json.load(f)
            
    score = 0
    results = []
    len_data = len(pred_data)
    
    for index, item in enumerate(tqdm(pred_data, desc="Processing items")):
        question = item['query']
        answer = item['answer']
        pred = item['pred']
        
        prompt_template = prompt.format(question=question, answer=answer, pred=pred)
        response = pipe(prompt_template, gen_config=gen_config)  

        response_text = response.text.strip().replace("'", "\"")
        
        try:
            response_dict = json.loads(response_text)
            score_i = float(response_dict['score'])
        except json.JSONDecodeError:
            print(f"Failed to parse response as JSON for item {index}: {response.text}")
            score_i = 0  
 
        score = score + score_i

        result = {
            "query": question,
            "pred": pred,
            "answer": answer,
            "score": score_i
        }
        results.append(result)
    
    average_score = score / len_data
    output_data = {
        "results": results,
        "average_score": average_score
    }
    
    with open(result_caption_question_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)


    print("evaluate caption question Done, average score: ", average_score)


     
if __name__ == '__main__':
    main()

