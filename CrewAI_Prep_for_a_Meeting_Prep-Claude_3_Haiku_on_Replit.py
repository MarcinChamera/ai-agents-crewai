'''
Max RPM for personal use is 5
'''

import os

# from langchain.agents import tool

# from langchain_community.tools import DuckDuckGoSearchRun

# @tool
# def duckduckgo_search(query: str):
#   """Search for a personal info or webpage based on the query."""
#   query = query.replace("'query':", "").replace("'q':", "")
#   return DuckDuckGoSearchRun().run(query)

# from exa_py import Exa

# class ExaSearchTool:

#   @tool
#   def search(query: str):
#     """Search for a webpage based on the query."""
#     return ExaSearchTool._exa().search(f"{query}",
#                                        use_autoprompt=True,
#                                        num_results=3)

#   @tool
#   def find_similar(url: str):
#     """Search for webpages similar to a given URL.
#     The url passed in should be a URL returned from `search`.
#     """
#     return ExaSearchTool._exa().find_similar(url, num_results=3)

#   @tool
#   def get_contents(ids: str):
#     """Get the contents of a webpage.
#     The ids must be passed in as a list, a list of ids returned from `search`.
#     """
#     ids = eval(ids)
#     contents = str(ExaSearchTool._exa().get_contents(ids))
#     print(contents)
#     contents = contents.split("URL:")
#     contents = [content[:1000] for content in contents]
#     return "\n\n".join(contents)

#   def tools():
#     return [
#         ExaSearchTool.search, ExaSearchTool.find_similar,
#         ExaSearchTool.get_contents
#     ]

#   def _exa():
#     return Exa(api_key=os.environ["EXA_API_KEY"])

from crewai_tools.tools import WebsiteSearchTool

web_search_tool = WebsiteSearchTool()

from textwrap import dedent
from crewai import Agent
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
  model="claude-3-haiku-20240307",
  verbose=True,
  temperature=0.3,
  anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))


## Agents
class MeetingPreparationAgents():

  def research_agent(self):
    return Agent(
        role='Research Specialist',
        goal=
        'Conduct thorough research on people and companies involved in the meeting',
        tools=[web_search_tool],
        backstory=dedent("""\
          As a Research Specialist, your mission is to uncover detailed information
          about the individuals and entities participating in the meeting. Your insights
          will lay the groundwork for strategic meeting preparation."""),
        llm=llm,
        max_rpm=2,
        verbose=True,
        memory=True)

  def industry_analysis_agent(self):
    return Agent(
        role='Tech Analyst',
        goal=
        'Analyze the current tech trends, challenges, and opportunities',
        tools=[web_search_tool],
        backstory=dedent("""\
          As a Tech Analyst, your analysis will identify key trends,
          challenges facing the tech, and potential opportunities that
          could be leveraged during the meeting for strategic advantage."""),
        llm=llm,
        max_rpm=2,
        verbose=True,
        memory=True)

  def meeting_strategy_agent(self):
    return Agent(
        role='Meeting Strategy Advisor',
        goal=
        'Develop talking points, questions, and strategic angles for the meeting',
        tools=[web_search_tool],
        backstory=dedent("""\
          As a Strategy Advisor, your expertise will guide the development of
          talking points, insightful questions, and strategic angles
          to ensure the meeting's objectives are achieved."""),
        llm=llm,
        max_rpm=1,
        verbose=True,
        memory=True)

  def summary_and_briefing_agent(self):
    return Agent(
        role='Briefing Coordinator',
        goal=
        'Compile all gathered information into a concise, informative briefing document',
        tools=[web_search_tool],
        backstory=dedent("""\
          As the Briefing Coordinator, your role is to consolidate the research,
          analysis, and strategic insights."""),
        llm=llm,
        max_rpm=1,
        verbose=True,
        memory=True)


## Tasks
from crewai import Task


class MeetingPreparationTasks():

  def research_task(self, agent, participants, context):
    return Task(description=dedent(f"""\
        Conduct comprehensive research on each of the individuals involved in the upcoming meeting. Gather information on recent news, achievements, professional background, and any relevant business activities. Today is April 3, 2024.

        Participants: {participants}
        Meeting Context: {context}"""),
                expected_output=dedent("""\
        A detailed report summarizing key findings about each participant, highlighting information that could be relevant for the meeting."""
                                       ),
                async_execution=True,
                agent=agent)

  def industry_analysis_task(self, agent, participants, context):
    return Task(description=dedent(f"""\
        Analyze the current tech trends, challenges, and opportunities
        relevant to the meeting's context. Consider market reports, recent
        developments, and expert opinions to provide a comprehensive
        overview of the tech landscape. Today is April 3, 2024.

        Participants: {participants}
        Meeting Context: {context}"""),
                expected_output=dedent("""\
        An insightful analysis that identifies major trends, potential
        challenges, and strategic opportunities."""),
                async_execution=True,
                agent=agent)

  def meeting_strategy_task(self, agent, context, objective):
    return Task(description=dedent(f"""\
        Develop strategic talking points, questions, and discussion angles
        for the meeting based on the research and industry analysis conducted

        Meeting Context: {context}
        Meeting Objective: {objective}"""),
                expected_output=dedent("""\
        Complete report with a list of key talking points, strategic questions
        to ask to help achieve the meetings objective during the meeting."""),
                agent=agent)

  def summary_and_briefing_task(self, agent, context, objective):
    return Task(description=dedent(f"""\
        Compile all the research findings, industry analysis, and strategic
        talking points into a concise, comprehensive briefing document for
        the meeting.
        Ensure the briefing is easy to digest and equips the meeting
        participants with all necessary information and strategies.

        Meeting Context: {context}
        Meeting Objective: {objective}"""),
                expected_output=dedent("""\
        A well-structured briefing document that includes sections for
        participant bios, industry overview, talking points, and
        strategic recommendations."""),
                agent=agent)


## Crew
from crewai import Crew

tasks = MeetingPreparationTasks()
agents = MeetingPreparationAgents()

participants = "Andrew Ng, Andrej Karpathy"
context = "Comptetitive analysis of 'AI Agents' technology"
objective = "Convince participants that the company should purchase Agents of Tomorrow, which is a startup that created a promising framework for orchestrating AI Agents"

# Create Agents
researcher_agent = agents.research_agent()
industry_analyst_agent = agents.industry_analysis_agent()
meeting_strategy_agent = agents.meeting_strategy_agent()
summary_and_briefing_agent = agents.summary_and_briefing_agent()

# Create Tasks
research = tasks.research_task(researcher_agent, participants, context)
industry_analysis = tasks.industry_analysis_task(industry_analyst_agent,
                                                 participants, context)
meeting_strategy = tasks.meeting_strategy_task(meeting_strategy_agent, context,
                                               objective)
summary_and_briefing = tasks.summary_and_briefing_task(
    summary_and_briefing_agent, context, objective)

meeting_strategy.context = [research, industry_analysis]
summary_and_briefing.context = [research, industry_analysis, meeting_strategy]

# Create Crew responsible for Copy
crew = Crew(agents=[
    researcher_agent, industry_analyst_agent, meeting_strategy_agent,
    summary_and_briefing_agent
],
            tasks=[
                research, industry_analysis, meeting_strategy,
                summary_and_briefing
            ])

print("Kickoff!")
result = crew.kickoff()

# Print results
print("\n\n################################################")
print("## Here is the result")
print("################################################\n")
print(result)

with open("result.txt", "w") as f:
  f.writelines(result)
