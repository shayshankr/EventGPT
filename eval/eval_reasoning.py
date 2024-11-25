from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
from tqdm import tqdm
import json


prompt = """Role and Objective:
You are an advanced evaluation chatbot assigned to assess the reasoning quality of generative outputs in image-based question-answer pairs, focusing on rigorous standards of logical precision and depth.

Your Task:
Compare the Predicted Answer with the Correct Answer and evaluate its reasoning quality. Emphasize strict adherence to logical coherence, completeness, accuracy, depth, and contextual relevance. Only minor rephrasings with exact reasoning intent should be accepted. Close approximations and alternative approaches are not permitted unless they are equally rigorous and convey identical depth and accuracy as the Correct Answer.

Instructions:

	•	Evaluate Logical Coherence: Ensure the Predicted Answer follows a strictly logical progression, maintaining a coherent and natural flow. Minor logical gaps or vague connections should not be accepted.
	•	Check Completeness: Verify that the Predicted Answer fully addresses all critical elements of the question. Partial answers or minor omissions are unacceptable.
	•	Assess Correctness: Confirm that the reasoning is entirely accurate, without any errors that might impact logic or understanding. Even minor inaccuracies should reduce the score.
	•	Consider Depth: Determine if the Predicted Answer demonstrates substantial depth, including nuanced details that reflect a deep understanding of underlying concepts. Superficial or oversimplified explanations should significantly lower the score.
	•	Ensure Relevance: Ensure that the reasoning is directly and precisely aligned with the question and image context. Irrelevant or tangential points are unacceptable, and each aspect of the answer should reinforce the answer’s relevance to the question and image.

Evaluation Criteria (Reasoning Score):
Provide a single evaluation score reflecting the Predicted Answer’s reasoning quality based on the above criteria. Use decimal scores (e.g., 4.3, 3.6) to capture fine distinctions in reasoning quality.

Score Guide:

	•	5: Outstanding reasoning—entirely logical, thorough, correct, and demonstrates deep understanding, highly relevant to the question and image. Minor rephrasings are allowed only if they convey identical reasoning.
	•	4-4.9: Strong reasoning—mostly logical, accurate, and relevant, with only minimal gaps. Depth and coherence should be high, though not perfect.
	•	3-3.9: Adequate reasoning—generally logical and mostly accurate but includes noticeable gaps or simplifications. May lack depth or show minor inaccuracies that weaken coherence.
	•	2-2.9: Flawed reasoning—contains significant logical issues, major inaccuracies, or lacks critical elements, resulting in poor coherence and insufficient depth.
	•	1-1.9: Very weak reasoning—largely incorrect, illogical, or irrelevant, with almost no coherence or depth.
	•	0-0.9: No reasoning provided or entirely unrelated to the question and image, lacking any logical or relevant response.

Evaluation Format:
Your evaluation should be in the following format:

	•	Question: {question}
	•	Correct Answer: {answer}
	•	Predicted Answer: {pred}

Your response should be a Python dictionary formatted as a string with the key ‘score’ and a floating-point value between 0 and 5, where 5 represents the highest reasoning quality.

Example Output Format:
{{'score': 3.5}}

Provide only the Python dictionary string, without additional explanations or comments."""

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

    pred_json_file_path = "/data/SyL/EventChat/eval/pred_results/reasoning_train_3_result.json"
    result_caption_question_file_path = "/data/SyL/EventChat/eval/eval_results/reasoning_train_3_eval.json"
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

    