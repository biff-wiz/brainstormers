import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from utils import parse_bullet_points, TreeNode, print_tree, InitialIdeaChain

mm_expand_idea_prompt = ChatPromptTemplate.from_template("""You are a clever idea expansion assistant that helps people expand one idea into 5 other related ideas. The resulting ideas should be diverse, detailed, developed, precise and significant. The ideas should not be redundant and repetitive, be creative and unique. The ideas must be formatted in the form of bullet points without titles and without bold text.
Idea to expand:{idea}
List of 5 bullet points ideas:""")

# Main loop
# Initialize the root node with the user's query

def bmm(user_query, llm):
    initial_idea_chain = InitialIdeaChain(llm)
    mm_expand_idea_chain = mm_expand_idea_prompt | llm | parse_bullet_points
    
    root = TreeNode(user_query)
    progress_bar = st.progress(0, "Generating initial ideas...")
    
    # Generate 10 initial ideas (33% of progress)
    initial_ideas = initial_idea_chain.invoke({"query": user_query})
    progress_bar.progress(0.33, "Expanding first level ideas...")
    
    # Add and expand each initial idea (66% of progress)
    for i, idea in enumerate(initial_ideas):
        child_node = TreeNode(idea)
        root.add_child(child_node)
        expanded_ideas = mm_expand_idea_chain.invoke({"idea": idea})
        
        # Add expanded ideas and generate final level (66% to 100%)
        for expanded_idea in expanded_ideas:
            grandchild_node = TreeNode(expanded_idea)
            child_node.add_child(grandchild_node)
            further_expanded_ideas = mm_expand_idea_chain.invoke({"idea": expanded_idea})
            
            # Add each further expanded idea as a child of the current expanded idea
            for further_expanded_idea in further_expanded_ideas:
                great_grandchild_node = TreeNode(further_expanded_idea)
                grandchild_node.add_child(great_grandchild_node)
                
        progress = 0.33 + (0.67 * (i + 1) / len(initial_ideas))
        progress_bar.progress(progress, "Expanding deeper level ideas...")

    progress_bar.progress(1.0, "Complete!")
    # Print the tree
    return print_tree(root)