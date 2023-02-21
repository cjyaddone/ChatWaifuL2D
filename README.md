![cover](readme/chatwaifu.png)

[中文](README.md "中文") [English](eng-README.md "English") [日本語](jp-README.md "日本語")

<p align="center">
	<img alt="GitHub" src="https://img.shields.io/github/license/cjyaddone/ChatWaifu?color=red">
	<img src="https://img.shields.io/badge/Python-3.7|8|9|10-green" alt="PYTHON" >
  	<a href="https://app.fossa.com/projects/git%2Bgithub.com%2Fcjyaddone%2FChatWaifu?ref=badge_small" alt="FOSSA Status"><img src="https://app.fossa.com/api/projects/git%2Bgithub.com%2Fcjyaddone%2FChatWaifu.svg?type=small"/></a>
</p>

#


> ### 这是一个基于TTS+VITS的ChatGPT语音对话程序!

效果演示BiliBIli:[ChatGPT+Galgame 与老婆对话！](https://b23.tv/yw48MVs "BiliBili")

**当前支持功能：**
* [x] ChatGPT的对话聊天
* [x] 回答转语音
* [x] 多角色语音
* [x] 语音识别对话 (研发了一款真正人性化的智能语音Q宝
* [x] 对接Live2D的Web版本
* [x] [对接Marai机器人](https://github.com/MuBai-He/ChatWaifu-marai)
* [x] [使用gpt3官方api，并支持cuda加速的版本（当前仅源码](https://github.com/cjyaddone/ChatWaifu-API)

# 运行方法
#### 确保已安装Chrome浏览器，并升级到最新的108版本
#### 如果您在国内，需要使用vpn
#### 下载并解压最新的Release
#### 运行ChatWaifuL2D.exe安装程序，建议安装在C盘外的目录
#### 运行ChatWaifuL2D中的ChatWaifuServer.exe
#### 运行ChatWaifuL2D/Game下的 ChatWaifu.exe

# 如何获取Token
#### 在浏览器登入https://chat.openai.com
#### 按F12进入开发控制台
#### 找到 应用程序 -> cookie -> __Secure-next-auth.session-token
![](readme/token.png)
#### 将值复制进入游戏并回车

## <span id="915">6.鸣谢：</span>
- [MoeGoe_GUI]https://github.com/CjangCjengh/MoeGoe_GUI
- [Pretrained models]https://github.com/CjangCjengh/TTSModels
- [PyChatGPT]https://github.com/terry3041/pyChatGPT
