import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

query_text = """Task:

Generate high-quality Visual Question Answering (VQA) examples by analyzing an image and providing a diverse set of question-and-answer pairs that explore various aspects of the visual scene.
Input:

You will be provided with an image containing a complex visual scene. This scene may involve multiple objects, interactions, environments, and contexts. Pay close attention to all details in the image, such as objects, actions, relationships, background, lighting, settings, and overall atmosphere.

Instructions to the Model:

Based on the content of the image, generate 3-5 natural language question-and-answer pairs that require deep understanding and reasoning. The questions should vary in complexity and cover different types of inquiries, including overall scene interpretation, object recognition, interactions, hypothetical reasoning, and detailed object attributes.

Requirements for the Questions:

	•	Target different levels of understanding, including the overall scene, individual objects, actions, relationships, and interactions.
	•	Use natural, conversational language; mix shorter, fact-based questions with more complex, reasoning-based ones.
	•	Include questions that explore the setting, mood, and context, as well as specific object details and actions.
	•	Encourage questions that synthesize multiple visual elements or require logical connections between objects and events.
	•	Vary the focus of the questions: some should address specific objects or actions, while others should target the scene’s overall narrative or atmosphere.

Requirements for the Answers:

	•	Provide accurate and contextually relevant answers based on visible content or reasonable inferences from the image.
	•	Ensure answers range from short, factual responses to more detailed explanations, depending on the complexity of the question.
	•	Avoid introducing information that cannot be inferred or is not present in the image.

Special Attention:

	•	Generate 3-5 question-and-answer pairs that vary in length, complexity, and focus.
	•	Aim for diversity in question types, including scene-level analysis, object-specific inquiries, hypothetical scenarios, and action-related reasoning.
	•	Maintain a balance between detailed, long responses and concise, direct answers.

Follow the format below:
Question1: ...
Answer1:...

Question2: ...
Answer2:...

...
"""

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

def main():
    pipe = pipeline('path/model/Qwen2-VL-72B-Instruct',
                    backend_config=TurbomindEngineConfig(tp=4))

    prompt = """As an expert competition question designer, your task is to create 5 multiple-choice question with 4 answer options based on a provided image. The question should focus on specific scenarios such as road traffic, autonomous driving, or knowledge-based quizzes. Ensure that the correct answer stands out with clear superiority, and that the distractor options are plausible yet distinctly less accurate or relevant. The question should test critical understanding and practical application in the chosen scenario, emphasizing clarity and precision."""
    format = """Please follow the format below: 
    Query: ...
    A)...
    B)...
    C)...
    D)...
    Answer: ..."""
    prompt = prompt + format
    image = load_image('path/image')
    gen_config = GenerationConfig(temperature=0.8)
    response = pipe((prompt, image), gen_config=gen_config)

    print(response.text)

if __name__ == '__main__':
    main()
	