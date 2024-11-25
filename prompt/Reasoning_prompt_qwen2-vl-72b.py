"""
### Task Overview:

You are an advanced multimodal model with the ability to perform complex reasoning by integrating both visual and textual information. In this task, you will analyze symbolic representations of visual scenes and generate sophisticated Query-Answer pairs that require deep reasoning. These pairs should focus on inferences, causal relationships, and predictions based on the visual cues provided.

### Task Steps:

1.	Scene Description: A symbolic representation of a scene will be provided, describing objects, their attributes, spatial positions, and relationships.
2.	Question Generation: Create a question that involves multi-step reasoning. The question should focus on deducing relationships, predicting outcomes, or making causal inferences from the scene.
3.	Answer Generation: Provide a clear, well-reasoned answer to the question, explaining the inference process or reasoning behind the outcome.

### Requirements for the Question:

•	The question should require deeper reasoning and not just simple observation.
•	It must involve the relationships or dynamics between objects and potentially future or hypothetical outcomes.
•	Ensure that the question prompts the model to think beyond just recognition or identification of objects.
•   Please refrain from mentioning any elements or questions related to color.

### Example:

Scene Description (Symbolic Representation):

•	Object 1: Red ball, located on a wooden table.
•	Object 2: Blue book, placed to the left of the red ball.
•	Object 3: Glass of water, sitting to the right of the red ball.
•	Relationship: The red ball is larger than the glass of water, and the blue book is open.

Generated Question:
“If the table is accidentally knocked over, what will happen to the objects on it? Predict the sequence of events and explain the reasoning behind it.”

Generated Answer:
“The red ball, blue book, and glass of water will all fall as the table collapses. The glass is likely to break, spilling water, while the book could get wet. The ball, being round, would roll away from the scene, while the book and glass remain near the table’s original position.”

### Output Format:

Provide the questions and answers in this format:

    question: [Your question here]
    answer: [Your answer here]

    question: [Your question here]
    answer: [Your answer here]

    question: [Your question here]
    answer: [Your answer here]

    question: [Your question here]
    answer: [Your answer here]

    question: [Your question here]
    answer: [Your answer here]
"""