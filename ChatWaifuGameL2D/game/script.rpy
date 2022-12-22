# 游戏的脚本可置于此文件中。

# 声明此游戏使用的角色。颜色参数可使角色姓名着色。

define e = Character("Hiyori")
define config.gl2 = True

image hiyori = Live2D("Resources/hiyori", base=.6, loop = True, fade=True)

init python:
    import socket
    import time
    renpy.block_rollback()
    ip_port = ('127.0.0.1', 9000)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect(ip_port)


# 游戏在此开始。

label start:
    $ renpy.block_rollback()
    # 显示一个背景。此处默认显示占位图，但您也可以在图片目录添加一个文件
    # （命名为 bg room.png 或 bg room.jpg）来显示。

    #scene bg library

    # 显示角色立绘。此处使用了占位图，但您也可以在图片目录添加命名为
    # eileen happy.png 的文件来将其替换掉。
    show hiyori m01

    #show eileen happy

    # 此处显示各行对话。


    python:
        token = renpy.input("让我们开始吧，请输入ChatGPT的Token")
        client.send(token.encode())
    
    e "Token已经收到，我们进入下一步吧"

    menu languageChoice:
        e "请选择一种输出语言"

        "中文":
            #block of code to run
            python:
                client.send(("0").encode())
            jump modelChoiceCN
        "日本語":
            #block of code to run
            python:
                client.send(("1").encode())
            jump modelChoiceJP

    return    
        
label modelChoiceCN:
    menu CNmodelChoice:
        e "我们来选择一个角色作为语音输出"

        "綾地寧々":
            #block of code to run
            python:
                client.send(("0").encode())
            jump talk
        "在原七海":
            #block of code to run
            python:
                client.send(("1").encode())
            jump talk
        "小茸":
            #block of code to run
            python:
                client.send(("2").encode())
            jump talk
        "唐乐吟":
            #block of code to run
            python:
                client.send(("3").encode())
            jump talk
    

label modelChoiceJP:
    menu JPmodelChoice:
        e "我们来选择一个角色作为语音输出"

        "綾地寧々":
            #block of code to run
            python:
                client.send(("0").encode())
            jump talk
        "因幡めぐる":
            #block of code to run
            python:
                client.send(("1").encode())
            jump talk
        "朝武芳乃":
            #block of code to run
            python:
                client.send(("2").encode())
            jump talk
        "常陸茉子":
            #block of code to run
            python:
                client.send(("3").encode())
            jump talk
        "ムラサメ":
            #block of code to run
            python:
                client.send(("4").encode())
            jump talk
        "鞍馬小春":
            #block of code to run
            python:
                client.send(("5").encode())
            jump talk
        "在原七海":
            #block of code to run
            python:
                client.send(("6").encode())
            jump talk
   
    
label talk:
    $ renpy.block_rollback()
    show hiyori m02
    python:
        message = renpy.input("你：")
        client.send(message.encode())
        data = bytes()
    jump checkRes


label checkRes:
    $ renpy.block_rollback()
    e "..."
    show hiyori m03

    python:
        client.setblocking(0)
        try:
                data = client.recv(1024)
        except:
                data = bytes()
                client.setblocking(1)
    
    if(len(data) > 0):
        $ response = data.decode()
        jump answer
    else:
        e "......"
        jump checkRes

        


label answer:
    show hiyori talking
    voice "/audio/test.ogg"
    e "[response]"
    voice sustain
    
    jump talk
