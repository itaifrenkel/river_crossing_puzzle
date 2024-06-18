import os


def llm():
    from dotenv import load_dotenv
    load_dotenv()
    if os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME']:
        return _azure_llm()
    else:
        return _openai_llm()


def _openai_llm():
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model_name="gpt-4-0613")
    return model


def _azure_llm():
    from langchain_openai import AzureChatOpenAI
    model = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        openai_api_version='2024-02-01',
    )
    return model
