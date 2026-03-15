import arxiv
import time

# --- Configuration ---
# Use a more explicit query targeting title or abstract
# Searching for "Text-to-SQL" in either title or abstract
search_query = '(ti:"Text-to-SQL" OR abs:"Text-to-SQL") OR (ti:Text AND ti:SQL) OR (abs:Text AND abs:SQL)' #'ti:"Text-to-SQL" OR abs:"Text-to-SQL"'
max_results = 200 # You can adjust this to get more results

# --- Search ---
print(f"Searching arXiv for '{search_query}'...")
client = arxiv.Client()

try:
    # Create the search object with the explicit query and sorting
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance # Relevance is usually best for keyword search
        # You can try SubmittedDate again after confirming results appear with Relevance
        # sort_by=arxiv.SortCriterion.SubmittedDate
    )

    paper_list = []
    # Iterate through the results. Using list() here fetches all results up to max_results.
    results = list(client.results(search))

    if not results:
        print("\nNo results found for the specified query.")
    else:
        print(f"\nFound {len(results)} papers matching the query.")
        print("--- Paper List (Title and ID) ---")
        for result in results:
            paper_list.append({
                'id': result.entry_id.split('/')[-1],
                'title': result.title,
                'published': result.published
            })
            print(f"Title: {result.title}")
            print(f"ID: {result.entry_id.split('/')[-1]}")
            print(f"Published: {result.published.strftime('%Y-%m-%d')}")
            print("-" * 20)
            # Add a small delay between processing results if needed, though not strictly necessary here
            # time.sleep(0.1)

except Exception as e:
    print(f"An error occurred during arXiv search: {e}")
    print("Please check your internet connection and ensure the 'arxiv' library is installed correctly.")


# The 'paper_list' variable now contains a list of dictionaries with paper info.
# You can now copy the IDs from the output list to use in build_rag_index.py
