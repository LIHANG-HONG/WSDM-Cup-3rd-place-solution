class PromptManager(object):
    preference_prompt_predict_template = '''<|im_start|>system<|im_sep|>You are a highly skilled AI assistant. You will be provided a request from user and several responses from different assistants. Your job is to judge which response is the best. Only answer the letter of the best response.<|im_end|><|im_start|>user<|im_sep|><Request>
{prompt}
</Request>

<Language>
{language}
</Language>

{responses}<|im_end|><|im_start|>assistant<|im_sep|>'''

    preference_prompt_train_template = preference_prompt_predict_template + '{answer}<|im_end|>'

    response_template = '''<Response_{choice}>
{response}
</Response_{choice}>\n'''

    sep = '<|im_start|>assistant<|im_sep|>'
