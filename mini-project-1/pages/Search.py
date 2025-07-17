import streamlit as st
from vector.search import SemanticSearcher, format_search_result
from typing import Optional

def initialize_searcher() -> Optional[SemanticSearcher]:
    """Initialize the semantic searcher with error handling"""
    try:
        return SemanticSearcher()
    except Exception as e:
        st.error(f"Failed to initialize search: {str(e)}")
        st.info("Make sure you have initialized the database and stored some chunks first.")
        return None

def main():
    st.title("🔍 Semantic Search")
    st.write("""
    Ask a question and I'll find the most relevant information from the FAQ documents.
    The search uses semantic similarity to find the best matches, even if the exact words don't match.
    """)
    
    # Initialize searcher
    searcher = initialize_searcher()
    if not searcher:
        return
        
    # Search interface
    with st.form("search_form"):
        query = st.text_input("Enter your question:")
        num_results = st.slider("Number of results", min_value=1, max_value=10, value=3)
        submitted = st.form_submit_button("Search")
        
        if submitted and query:
            with st.spinner("Searching..."):
                results = searcher.search(query, limit=num_results)
                
                if not results:
                    st.warning("No relevant results found. Try rephrasing your question.")
                else:
                    st.success(f"Found {len(results)} relevant matches!")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i}: {result['question'][:100] or 'No question available'}"):
                            st.markdown(format_search_result(result))
                            
                            # Show full content in a code block if needed
                            if st.checkbox(f"Show full content for result {i}"):
                                st.code(result['content'])

if __name__ == "__main__":
    main() 