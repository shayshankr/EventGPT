from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import json
from tqdm import tqdm


prompt = """Role and Objective:
You are an intelligent chatbot designed to evaluate the detail orientation of generative outputs for image-based question-answer pairs.

Your Task:
Compare the Predicted Answer with the Correct Answer and assess its level of detail, focusing on completeness and specificity. Consider synonyms, paraphrases, and close approximations as valid matches, even if minor variations occur.

Instructions:

• Check Completeness: Ensure the Predicted Answer generally covers the major points from the image, including key aspects that provide essential context. Minor omissions are acceptable as long as the main content is largely conveyed.

• Evaluate Specificity: Confirm that the Predicted Answer includes specific details tied to the image, allowing for varied descriptions, synonyms, or rephrased elements that still accurately convey the intended meaning.

• Allow Synonyms, Paraphrases, and Close Matches: Accept synonyms, paraphrases, or approximate matches to details in the Correct Answer. Minor rewording, alternative phrasing, or equivalent descriptions should be considered as aligned with the Correct Answer if they capture the general essence.

Evaluation Criteria (Detail Orientation Score):
Provide a single evaluation score that reflects the level of detail orientation in the Predicted Answer, considering both completeness and specificity. You may use decimal scores (e.g., 4.5, 3.7) to reflect nuanced distinctions in detail orientation.

Score Guide:
• 5: Perfectly detailed, covering all major points and specific details with accuracy, even if phrased differently. Synonyms and alternative phrasing match the intent fully.

• 4-4.9: Highly detailed, capturing most key points and specific details. Minor omissions or rephrasing that still convey the essence of the Correct Answer are acceptable.

• 3-3.9: Moderately detailed, covering a substantial number of key points and specific details. Noticeable omissions are present, but main aspects are still represented.

• 2-2.9: Limited detail, missing several key points or specific details, with parts of the answer that are vague or incomplete.

• 1-1.9: Minimal detail, capturing few relevant points, and showing limited alignment with the major content of the Correct Answer.

• 0-0.9: No alignment with relevant details from the image content, with the Predicted Answer missing key aspects entirely.

Evaluation Format:
Provide your evaluation in the following format:

• Question: {question}
• Correct Answer: {answer}
• Predicted Answer: {pred}

Your response should be a Python dictionary in the form of a string with the key ‘score’ and a floating-point value between 0 and 5, where 5 indicates the highest level of detail orientation.

Example Output Format:
{{'score': 4.3}}

Do not include any other text or explanations beyond the Python dictionary string."""

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

    pred_json_file_path = "/data/SyL/EventChat/ablation/no-st/pred/event_vqa_result.json"
    result_caption_question_file_path = "/data/SyL/EventChat/ablation/no-st/result/event_vqa_eval.json"
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

    