from langgraph.graph import StateGraph, END
import ollama
from typing import TypedDict, List

class Mystate(TypedDict):
    retreived_chunks: List[str]
    rephrased_chunks: List[str]
    model: str
    role: str

class Chatbot_response:
    def __init__(self):
        self.graph=self.build_graph()

    def role_selection(self, state_data: Mystate)-> Mystate:
        user_input_role=state_data.get("role")

        role_map={
            "dev": "Software Developer",
            "ba":"Business Analyst",
            "Tester":"Software Quality engineer",
            "management":"Organization Leadership"
        }

        selected_role=role_map.get(user_input_role.lower())
        state_data["role"]=selected_role
        print(f"Selected role is : {selected_role}")
        return state_data

    def rephrase_chunks(self,state_data: Mystate)->Mystate:
        few_shot_examples = {
        "Business Analyst": [
            {
                "input": "AI tools improve data analysis speed and accuracy.",
                "output": "As a Business Analyst, this highlights how AI enhances data-driven decision-making, enabling quicker and more accurate insights for business strategy."
            }
        ],
        "Software Developer": [
            {
                "input": "AI tools improve data analysis speed and accuracy.",
                "output": "As a Software Developer, AI tools like ML models and data pipelines can be implemented to automate data analysis tasks, reducing manual processing."
            }
        ],
        "Project Manager": [
            {
                "input": "AI tools improve data analysis speed and accuracy.",
                "output": "As a Project Manager, this means teams can deliver results faster and with more precision, improving overall project timelines and outcomes."
            }
        ]
    }
        chunks=state_data["retreived_chunks"]
        role=state_data.get("role","Business Analyst")
        model=state_data.get("model","mistral")

        examples=few_shot_examples.get(role,[])
        example_text=""
        for example in examples:
            example_text+=f"\nExample :\nOriginal: {example['input']}\n rephrased for {role}: {example['output']}\n"
        
        rephrased=[]
        for chunk in chunks:
            prompt=f""" You are a {role}. Rephrase the following content to highlight its relevance to your role's priorities such as business impact, technical implementation or decision making.
            Content:
            \"\"\"{chunk}"\"\"\"
            Rephrased for {role}:"""
            response=ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
            rephrased.append(response['message']['content'])
        state_data["rephrased_chunks"]=rephrased
        return state_data

    def print_results(self, state_data: Mystate) -> Mystate:
        for index, text in enumerate(state_data["rephrased_chunks"],1):
            print(f"\n chunk {index}:\n {text}")
        return state_data

    def build_graph(self):
        builder=StateGraph(Mystate)

        builder.add_node("RoleSelection", self.role_selection)
        builder.add_node("RephraseChunks",self.rephrase_chunks)
        builder.add_node("PrintResult",self.print_results)

        builder.set_entry_point("RoleSelection")
        builder.add_edge("RoleSelection","RephraseChunks")
        builder.add_edge("RephraseChunks","PrintResult")
        builder.add_edge("PrintResult",END)

        graph=builder.compile()
        return graph
    
    def run(self,input_state: Mystate)->Mystate:
       return self.graph.invoke(input_state)

'''
# Usage
if __name__ == "__main__":
    input_state = {
        "retreived_chunks": [
            "Artificial intelligence in business is the use of AI tools such as machine learning natural language processing and computer vision to optimize business functions boost employee productivity and drive business value.",
            "Elsevier's Five Responsible AI Principles We take action to prevent the creation or reinforcement of unfair bias We can explain how our solutions work We create accountability through human oversight"
        ],
        "role": "dev",
        "model": "mistral"
    }
    graph_runner = chatbot_response()
    result_state = graph_runner.run(input_state)
'''

