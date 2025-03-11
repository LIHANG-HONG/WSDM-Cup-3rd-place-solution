class PromptManager(object):
    preference_prompt_predict_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a highly skilled AI assistant. You will be provided a request from user and several responses from different assistants. Your job is to judge which response is the best. Only answer the letter of the best response.<|eot_id|><|start_header_id|>user<|end_header_id|>

<Request>
{prompt}
</Request>

<Language>
{language}
</Language>

{responses}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
    preference_prompt_train_template = preference_prompt_predict_template + '{answer}<|eot_id|>'
    response_template = '''<Response_{choice}>
{response}
</Response_{choice}>\n'''
    sep = '<|start_header_id|>assistant<|end_header_id|>\n\n'
