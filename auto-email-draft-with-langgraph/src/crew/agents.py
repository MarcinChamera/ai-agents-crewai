from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.get_thread import GmailGetThread
from langchain_community.tools.tavily_search import TavilySearchResults

from textwrap import dedent
from crewai import Agent
from langchain_openai import ChatOpenAI
from .tools import CreateDraftTool

class EmailFilterAgents():
	def __init__(self):
		self.gmail = GmailToolkit()
		self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)

	def email_filter_agent(self):
		return Agent(
			role='Senior Email Analyst',
			goal='Filter out non-essential emails like newsletters and promotional content',
			backstory=dedent("""\
				As a Senior Email Analyst, you have extensive experience in email content analysis.
				You are adept at distinguishing important emails from spam, newsletters, and other
				irrelevant content. Your expertise lies in identifying key patterns and markers that
				signify the importance of an email."""),
			llm=self.llm,
			verbose=True,
			allow_delegation=False
		)

	def email_action_agent(self):

		return Agent(
			role='Email Action Specialist',
			goal='Identify action-required emails and compile a list of their IDs',
			backstory=dedent("""\
				With a keen eye for detail and a knack for understanding context, you specialize
				in identifying emails that require immediate action. Your skill set includes interpreting
				the urgency and importance of an email based on its content and context."""),
			tools=[
				GmailGetThread(api_resource=self.gmail.api_resource),
				TavilySearchResults()
			],
			llm=self.llm,
			verbose=True,
			allow_delegation=False,
		)

	def email_response_writer(self):
		return Agent(
			role='Email Response Writer',
			goal='Draft responses to action-required emails',
			backstory=dedent("""\
				Your name is John Doe.
				You are a skilled writer, adept at crafting clear, concise, and effective email responses.
				Your strength lies in your ability to communicate effectively, ensuring that each response is
				tailored to address the specific needs and context of the email."""),
			tools=[
				TavilySearchResults(),
				GmailGetThread(api_resource=self.gmail.api_resource),
				CreateDraftTool.create_draft
			],
			llm=self.llm,
			verbose=True,
			allow_delegation=False,
		)