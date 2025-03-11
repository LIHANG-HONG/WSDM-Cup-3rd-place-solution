class PromptManager(object):
    preference_prompt_predict_template = '''<|im_start|>system
You are a highly skilled AI assistant. You will be provided a request from user and several responses from different assistants. Your job is to judge which response is the best. Only answer the letter of the best response.<|im_end|>
<|im_start|>user
<Request>
{prompt}
</Request>

<Language>
{language}
</Language>

{responses}<|im_end|>
<|im_start|>assistant
'''

    preference_prompt_train_template = preference_prompt_predict_template + '{answer}<|im_end|>\n'

    response_template = '''<Response_{choice}>
{response}
</Response_{choice}>\n'''

    sep = '<|im_start|>assistant\n'
