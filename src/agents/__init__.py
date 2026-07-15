"""계층형 멀티에이전트 재설계용 패키지.

LLM이 고정 블록(pipeline_exec의 고정 연산 화이트리스트)을 조립하는 대신 SQL/Python 코드를
직접 작성해 실행하도록 하는 신규 실행 계층이 여기에 산다. quant_trader(실거래 봇)와는
완전히 무관한 별도 모듈이며, pipeline_exec.py는 이 패키지에서 import/수정하지 않는다.
"""
