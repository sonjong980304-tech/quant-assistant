"""donegate — dart 완성도 게이트(Definition-of-Done gate) 러너.

done.yaml에 선언된 3축(기능/안정성/성능)을 command 경계로만 실행·판정한다.

분리 불변식(중요): 이 패키지는 dart 코드(`src/`)를 절대 import하지 않는다.
축과의 상호작용은 오로지 subprocess command 경계로만 이뤄진다(AC11). 이는
2번째 봇으로 공용화할 때 프레임워크/언어 종속을 없애고, dart-lock을 방지한다.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
