from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# --- Constants ---
APP_NAME = "code_pipeline_app"
USER_ID = "dev_user_01"
SESSION_ID = "pipeline_session_01"
GEMINI_MODEL = "gemini-2.0-flash-001"

# --- 1. 定義每個流水線階段的子 Agent ---

# 編碼 Agent：根據使用者的請求生成初始 Python 程式碼
code_writer_agent = LlmAgent(
    name="CodeWriterAgent",
    model=GEMINI_MODEL,
    instruction="""你是個程式碼撰寫 AI。
根據使用者的請求撰寫初始的 Python 程式碼。
僅輸出 *原始程式碼區塊*。""",
    description="根據使用者需求生成初始程式碼。",
    output_key="generated_code"
)

# 程式碼審查 Agent：檢查程式碼中的問題並給予回饋
code_reviewer_agent = LlmAgent(
    name="CodeReviewerAgent",
    model=GEMINI_MODEL,
    instruction="""你是個程式碼審查 AI。
請審查 session state 中 'generated_code' 欄位所提供的 Python 程式碼。
針對潛在錯誤、程式碼風格或可改進之處提供建設性回饋。
重點關注程式碼的可讀性與正確性。
僅輸出審查意見。""",
    description="審查程式碼並提供改進建議。",
    output_key="review_comments"
)

# 程式碼重構 Agent：根據審查意見優化程式碼
code_refactorer_agent = LlmAgent(
    name="CodeRefactorerAgent",
    model=GEMINI_MODEL,
    instruction="""你是個程式碼重構 AI。
請使用 session state 中 'generated_code' 欄位的原始 Python 程式碼，
以及 'review_comments' 欄位的審查意見，
進行程式碼重構，以解決意見中提出的問題並提升程式碼品質。
僅輸出 *最終重構後的程式碼區塊*。""",
    description="根據回饋對程式碼進行優化與重構。",
    output_key="refactored_code"
)

# 測試程式碼撰寫 Agent：根據重構後的程式碼自動生成對應的測試程式
test_writer_agent = LlmAgent(
    name="TestWriterAgent",
    model=GEMINI_MODEL,
    instruction="""你是個測試程式碼撰寫 AI。
請根據 session state 中 'refactored_code' 欄位提供的最終重構後 Python 程式碼，
撰寫對應的單元測試程式碼，使用 pytest 或 unittest 框架。
僅輸出 *測試程式碼區塊*。""",
    description="根據重構後程式碼生成單元測試。",
    output_key="test_code"
)

# --- 2. 建立順序執行的 Agent ---
code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[
        code_writer_agent,
        code_reviewer_agent,
        code_refactorer_agent,
        test_writer_agent
    ]
)

# 會話與執行器配置
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=code_pipeline_agent, app_name=APP_NAME, session_service=session_service)

# Agent 呼叫接口
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

root_agent = code_pipeline_agent