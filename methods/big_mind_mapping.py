from langchain_core.prompts import ChatPromptTemplate
from utils import parse_bullet_points, TreeNode, print_tree, InitialIdeaChain

mm_expand_idea_prompt = ChatPromptTemplate.from_template("""You are a clever idea expansion assistant that helps people expand one idea into 5 other related ideas. The resulting ideas should be diverse, detailed, developed, precise and significant. The ideas should not be redundant and repetitive, be creative and unique. The ideas must be formatted in the form of bullet points without titles and without bold text.
Idea to expand:{idea}
List of 5 bullet points ideas:""")







# Main loop
# Initialize the root node with the user's query

def bmm(user_query,llm):
    initial_idea_chain = InitialIdeaChain(llm)
    mm_expand_idea_chain = mm_expand_idea_prompt | llm | parse_bullet_points
     
    root = TreeNode(user_query)

    # Generate 10 initial ideas
    initial_ideas = initial_idea_chain.invoke({"query": user_query})

    # Add each initial idea as a child of the root
    for idea in initial_ideas:
        child_node = TreeNode(idea)
        root.add_child(child_node)

        # Expand each initial idea into 10 more ideas
        expanded_ideas = mm_expand_idea_chain.invoke({"idea": idea})

        # Add each expanded idea as a child of the current initial idea
        for expanded_idea in expanded_ideas:
            grandchild_node = TreeNode(expanded_idea)
            child_node.add_child(grandchild_node)

            # Expand each expanded idea into 10 more ideas
            further_expanded_ideas = mm_expand_idea_chain.invoke({"idea": expanded_idea})
            
            # Add each further expanded idea as a child of the current expanded idea
            for further_expanded_idea in further_expanded_ideas:
                great_grandchild_node = TreeNode(further_expanded_idea)
                grandchild_node.add_child(great_grandchild_node)

    # Print the tree
    return print_tree(root)
