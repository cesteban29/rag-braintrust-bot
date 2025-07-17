from braintrust import Eval
 
from autoevals import LLMClassifier


 # I need to save the context retrieved from the tool to the metadata of the task as context 
 # so that it can then be read and used for scoring from the context by the braintrust scorer.
no_apology = LLMClassifier(
    name="No apology",
    prompt_template="Does the response contain an apology? (Y/N)\n\n{{output}}",
    choice_scores={"Y": 0, "N": 1},
    use_cot=True,
)
 
Eval(
    "Say Hi Bot",  # Replace with your project name
    data=lambda: [
        {
            "input": "David",
            "expected": "Hi David",
        },
    ],  # Replace with your eval dataset
    task=lambda input: "Sorry " + input,  # Replace with your LLM call
    scores=[no_apology],
)