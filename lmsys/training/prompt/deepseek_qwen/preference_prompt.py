class PromptManager(object):
    preference_prompt_predict_template = '''<｜begin▁of▁sentence｜>You are a highly skilled AI assistant. You will be provided a request from user and several responses from different assistants. Your job is to judge which response is the best. Only answer the letter of the best response.<｜User｜><Request>
{prompt}
</Request>

<Language>
{language}
</Language>

{responses}<｜Assistant｜>'''

    preference_prompt_train_template = preference_prompt_predict_template + '{answer}<｜end▁of▁sentence｜>'

    response_template = '''<Response_{choice}>
{response}
</Response_{choice}>\n'''

    sep = '<｜Assistant｜>'
