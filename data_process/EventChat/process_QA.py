import uuid
import json

def transform_data(data):
    transformed_data = []
    
    for entry in data:
        qa_pairs = entry['QA_pair'].split('\n\n')
        for qa in qa_pairs:
            lines = qa.splitlines()
            if len(lines) >= 2 and "Question" in lines[0] and "Answer" in lines[1]:
                try:
                    question_text = lines[0].split(': ', 1)[1]
                    answer_text = lines[1].split(': ', 1)[1]

                    transformed_data.append({
                        "id": str(uuid.uuid4()),
                        "image": entry["image"],
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<image>\n{question_text}"
                            },
                            {
                                "from": "gpt",
                                "value": answer_text
                            }
                        ]
                    })
                except IndexError:
                    print(f"Error processing QA pair: {qa}")
            else:
                print(f"Skipped invalid QA pair: {qa}")
    
    return transformed_data

def read_json(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(output_file, data):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main(input_file, output_file):
    data = read_json(input_file)
    transformed_data = transform_data(data)
    write_json(output_file, transformed_data)

if __name__ == "__main__":
    input_file = "path/QA.json"   
    output_file = "path/instruction.json" 
    main(input_file, output_file)