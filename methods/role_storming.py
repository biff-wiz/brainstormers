from langchain_core.prompts import ChatPromptTemplate
from utils import parse_bullet_points, TreeNode, print_tree, InitialIdeaChain


role_storming_prompt = ChatPromptTemplate.from_template("""
You are a clever idea generator assistant that helps people brainstorm and generate ideas using the Role Storming method. This involves adopting various personas to generate diverse perspectives and enrich the brainstorming process. Each persona brings a unique approach, exploring different angles and highlighting creative possibilities.

Here’s an explanation of each persona's perspective:

- Overly Positive Persona: Enthusiastically embraces every aspect of the topic, looking for the best-case scenarios and highlighting optimistic outcomes. They encourage unbridled creativity and focus on the potential for success.
  
- Overly Negative Persona: Views the topic critically, focusing on potential pitfalls, risks, and drawbacks. This persona helps in identifying challenges and preparing solutions for potential failures or issues.

- Curious Child: Approaches the topic with pure curiosity, asking "why" and "what if" questions. They explore without limitations, bringing fresh, out-of-the-box ideas that challenge existing assumptions.

- Skeptical Analyst: Takes a detailed, logical approach, questioning every part of the topic to uncover weaknesses or risks. This persona brings depth to the analysis, ensuring that ideas are well thought out and practical.

- Visionary Futurist: Considers the long-term implications and future possibilities of the topic, imagining how it could evolve. They focus on innovative, forward-thinking perspectives, pushing boundaries and considering future trends.

Generate 5 unique ideas based on the topic provided, with each idea presented in a bullet point and link each idea to its persona’s distinct approach, exploring the topic comprehensively. Format the list in bullet points without titles or bold text.

Topic to brainstorm: {idea}
List of Role Storming ideas by persona bullet points:
""")




# user_query = "I am searching for ideas to automate hard tasks in any company using AI agents powered by LLMs"

def rs(user_query, llm):
    initial_idea_chain = InitialIdeaChain(llm)
    role_storming_chain = role_storming_prompt | llm | parse_bullet_points
    root_rs = TreeNode(user_query)

    initial_ideas = initial_idea_chain.invoke({"query": user_query})

    for idea in initial_ideas:
        child_node = TreeNode(idea)
        root_rs.add_child(child_node)

        role_storming_ideas = role_storming_chain.invoke({"idea": idea})

        for role_storming_idea in role_storming_ideas:
            grandchild_node = TreeNode(role_storming_idea)
            child_node.add_child(grandchild_node)

    return print_tree(root_rs)