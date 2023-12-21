'''
===========================================
        Module: Prompts collection
===========================================
'''
# Note: Precise formatting of spacing and indentation of the prompt template is important for Llama-2-7B-Chat,
# as it is highly sensitive to whitespace changes. For example, it could have problems generating
# a summary from the pieces of context if the spacing is not done correctly

system_prompt="Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"

qa_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""