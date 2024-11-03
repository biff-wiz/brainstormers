
from langchain_core.prompts import ChatPromptTemplate
from utils import parse_bullet_points, TreeNode, print_tree, InitialIdeaChain


reverse_brainstorming_prompt = ChatPromptTemplate.from_template("""
You are a perceptive problem-identification assistant that helps people analyze an idea by uncovering 5 potential issues or challenges it may encounter. The identified problems should be diverse, detailed, well-developed, precise, and significant. Avoid redundancy and repetition; ensure the problems are creative and unique. Present the problems in bullet points without titles and without bold text.

Idea to analyze: {idea}
List of 5 potential problems:
""")



# user_query = "I am searching for ideas to automate hard tasks in any company using AI agents powered by LLMs"
def rb(user_query,llm):
    initial_idea_chain = InitialIdeaChain(llm)
    reverse_brainstorming_chain = reverse_brainstorming_prompt | llm | parse_bullet_points
    root_rb = TreeNode(user_query)

    initial_ideas = initial_idea_chain.invoke({"query": user_query})

    for idea in initial_ideas:
        child_node = TreeNode(idea)
        root_rb.add_child(child_node)

        reverse_brainstorming_ideas = reverse_brainstorming_chain.invoke({"idea": idea})

        for reverse_brainstorming_idea in reverse_brainstorming_ideas:
            grandchild_node = TreeNode(reverse_brainstorming_idea)
            child_node.add_child(grandchild_node)

    return print_tree(root_rb)