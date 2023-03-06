# -*- coding: utf-8 -*-
from psychopy import visual,gui,core,event
from textwrap import fill
import os
import numpy as np
VERSION=3
def infobox(pos,fn=None):
    import datetime
    from psychopy import gui          

    
    myDlg = gui.Dlg(title='VP Info',pos=pos)    
    myDlg.addField('VP ID:',0)# subject id
    today=datetime.date.today()
    vls=['MS','LW']
    if not fn is None:
        myDlg.addField('Geschlecht:',choices=('mann','frau')) 
        myDlg.addText('Geburtsdatum')
        myDlg.addField('Tag',choices=range(1,32),initial=14)
        myDlg.addField('Monat',choices=range(1,13),initial=5)
        myDlg.addField('Jahr',choices=[2014,2015],initial=2014)
        myDlg.addText('Datenverwaltung')
        myDlg.addField('Versuchsleiter',choices=vls)
        myDlg.addField('Datasharing',initial=False)
    myDlg.show()#show dialog and wait for OK or Cancel
    if myDlg.OK:
        import numpy as np
        d=myDlg.data 
        if not fn is None and not d[0]==0:
            age=today-datetime.date(d[4],d[3],d[2])
            vlid=vls.index(d[5])
            sch=int(today.weekday()<3)
            vpinfo=[d[0],int(d[1]=='mann'),age.days,vlid,np.int32(d[6]),sch]
            strout= ('\n'+'{},'*(len(vpinfo)-1)+'{}').format(*(vpinfo))
            with open(fn,'a') as f: f.write(strout)
        #curd=str(datetime.datetime.today())
        #curd='-'+curd[:10]+'-'+curd[11:19];curd=curd.replace(':','-')
        return d[0]
    else:
        import sys
        sys.exit()
infoPath=os.path.dirname(os.path.realpath(__file__))+os.path.sep
vpinfofn='vpinfo.res'
vpid=infobox((0,0),fn=infoPath+vpinfofn)
    
def waitForKey(keys):
    done=False
    while not done:
        core.wait(0.05)
        for key in event.getKeys():
            if key in keys:
                return key
                
W=[u'möglich',u'unmöglich',u'sicher']#,u'wahrscheinlich',u'unwahrscheinlich']
B=[u'Sonne geht übermorgen auf',
u'Ampel wird irgendwann rot',
u'Ampel wird irgendwann blau',
u'Übermorgen regnet es',
u'Übermorgen schneit es',
u'Übermorgen hagelt es',
u'Übermorgen muss ich zur Schule',
u'Der 1.FC Köln gewinnt das nächste Spiel',
u'Heute Abend gibt es ein Feuerwerk',
u'Heute Abend gibt es Fischstäbchen']
L=['Mit [Leertaste] gehts weiter',['space'],-1]
JN=['[S]kip,[J]a, [N]ein, [K]eine Ahnung',['s','j','n','k'],-1]
JNH=['[S]kip,[J]a, [N]ein, [K]eine Ahnung, [H]ilfe',['s','j','n','k','h'],-1]
OR=['[E]rsteres\n[Z]weiteres\n[G]leich,[K]eine Ahnung,[S]kip',['s','e','z','g','k'],-1]
#################################################################
# EXPLAINING CLASSES AND ASKING FOR CLASS EXAMPLES
#################################################################
D=[[u"""
Ich habe einige Fragen für dich. Es ist kein Test, keine Prüfung. Es gibt kein richtig oder falsch. Sag mir einfach, was du meinst was richtig ist. 
"""]+L]

tmp0=u'''Zuerst möchte ich wissen ob du bestimmte Wörter verstehst.
1. Hast du das Wort MUSWU schon  mal gehört?
\t [nicht gehört] Alle nachfolgenden Unteraufgaben mit dem Wort weglassen
2. Kannst du sagen was man damit meint wenn man sagt etwas ist MUSWU?
\t [sagt nichts] ... wenn man sagt es ist MUSWU dass etwas passiert?
\t [keine Antwort/ nennt Beispiel] Wie würdest du das Wort einem kleinen Kind erklären?
3. Fällt dir ein Beispiel ein für etwas das MUSWU ist?
\t [sagt nichts] Wann würde man das sagen? Es ist möglich dass...?

'''   
D+=[[tmp0]+L]
#################################################################
# ASSIGNING EXAMPLES TO CLASSES
################################################################# 
tmp0=u'Ich habe jetzt einige Beispiele für dich. Du sagst mir ob das MUSWU ist oder nicht. '+ \
'Auf diese Seite legen wir Sachen die MUSWU sind und auf diese Seite den Rest.\n'+ \
'[Kärtchen vorlegen und vorlesen] Ist das MUSWU oder nicht?\n\n'+ \
'[wenn sicher oder unmöglich im anderen Sinne] korrigieren\n'+ \
'[als Antwort gilt] Kind sagt "MUSWU" oder "nicht MUSWU" oder Kind zeigt.\n'+ \
'\t"Möglich" ist NICHT gleich zu setzen mit "nicht unmöglich" - bitte nachfragen.\n'+ \
'Bei unmöglich und unwahrscheinlich den Kindern helfen doppelte Verneinung zu vermeiden. "Zeig mir einfach auf welche Seite das kommt."'''

#tmp1=u'{0}\n\nIst es {1}?'
D+=[[tmp0]+L]
#for j in range(len(B)):
#    D+=[[tmp1.format(B[j],W[i])]+JN]
#################################################################
# CLASS INTERSECTIONS
#################################################################
tmp=u'''Wir hatten Beispiele für Sachen die {0} und {1} sind. Kann es etwas geben das beides gleichzeitig ist? Das gleichzeitig {0} und {1} ist?'''
T=[]
for i in range(len(W)):
    for j in range(i+1,len(W)): 
        T+=[[tmp.format(W[i],W[j])]+JN]
assert(len(T)==3)
CIperm=np.random.permutation(len(T))
rCIperm=np.ones(len(T))*np.nan
for i in range(len(T)):
    D.append(T[CIperm[i]])
    rCIperm[CIperm[i]]=i
rCIperm=np.int32(rCIperm)
    
#################################################################
# EXPLAINING COMPARATOR AND EXAMPLES
#################################################################                
D+= [[u'''1.Kann man sagen dass etwas wahrscheinlicher ist als etwas anderes? Hast du das schon mal gehört?
\t[wenn nie gehört mit 1A-1B Beispiel helfen] Wenn man diese zwei Beispiele betrachtet [1A-1B vorlesen] kann man sagen, dass eins von diesen wahrschelicher ist als das andere? Welches?
2. Was meint man damit? Kannst du das erklären?
\t[keine Antwort] Man sagt dass etwas wahrscheinlicher ist als etwas anderes ...
3. Kannst du ein Beispiel nennen für etwas das wahrscheinlicher ist als etwas anderes?''']+L]
D+= [[u'Kann es Ereignisse geben die gleich wahrscheinlich sind?\n [H]ilfe falls Hilfestellung bei voriger Frage']+JNH]
#D+= [[u'Kannst du Beispiel für zwei Ereignise nennen die gleich wahrscheinlich sind?']+L] 
#################################################################
# DECIDING EXAMPLES
################################################################# 
D+= [[u'Ich habe wieder ein Paar Beispiele und ich möchte, dass du mir sagst welches wahrscheinlicher ist.'] +L]  
tmp=u'{0}\n\n\t\t\t\toder\n\n{1}\n\nWas ist wahrscheinlicher?\n[Antwort kommt. Dreieck hinlegen.] Das heißt X ist wahrscheinlicher als Y, oder? [Bestätigung abwarten]'
for i in [[3,4],[1,2],[3,5],[0,1],[4,5],[0,2]]:
    D+=[[tmp.format(B[i[0]],B[i[1]])]+OR]          
#################################################################
# TRANSITIVITY
#################################################################                
D+= [[u'''Stell dir vor, die Wissenschaftler haben ein neues Tier entdeckt. Es heißt Dando. Wissenschaftler haben herausgefunden dass Dando unterschiedliche Geräusche macht. 
Dando macht Nuf, Dando macht Bun und Dando macht Kup. Wissenschaftler haben auch herausgefunden dass Nuf wahrscheinlicher ist als Bun und dass Bun wahrscheinlicher ist als Kup. 
Was glaubst du, wie ist es dann wenn wir Nuf und Kup vergleichen? Ist eins von diesen wahrscheinlicher als das andere? Welches?\n\nWir Transitivität erwartet? Nuf w'er als Kup?''']+JN]
D+= [[u'''Dando und die Geräusche sind ausgedachte Sachen. Man kann sich auch andere drei Sachen vorstellen, so dass dieses wahrscheinlicher ist als dieses und dass dieses wiederum wahrscheinlicher ist als dieses.
Ist es in einem solchen Fall immer so wie du vorher bei Dando gesagt hast dass dieses wahrscheinlicher ist als dieses oder kann es zwischen diesen Beiden auch anders sein?\n\n Ist Transitivität zwingend?''']+JN]
#################################################################
# EQUIVALENCE CLASSES
#################################################################
D+= [[u'Stell dir vor wir haben hier Sachen die alle möglich sind? '+
    'Kann es zwei geben so dass eins wahrscheinlicher ist als das andere?\n'+
    'Alle gleich wahrscheinlich?']+JN]
D+= [[u'Stell dir vor wir haben hier Sachen die alle unmöglich sind? '+
    'Kann es zwei geben so dass eins wahrscheinlicher ist als das andere?\n'+
    'Alle gleich wahrscheinlich?']+JN]
D+= [[u'Stell dir vor wir haben hier Sachen die alle sicher sind? '+
    'Kann es zwei geben so dass eins wahrscheinlicher ist als das andere?\n'+
    'Alle gleich wahrscheinlich?']+JN]
#################################################################
# MINIMUM/MAXIMUM
#################################################################
D+= [[u'Stell dir vor wir suchen etwas das wahrscheinlicher ist als alle Sachen die wir bislang hatten.'+
    'Und wenn wir sowas finden dann suchen wir etwas das noch wahrscheinlicher ist. Kann man das so immer unendlich weiter machen oder gibt es da ein Ende?\n'+
    '\nHilfestellung: Du musst die Sachen nicht nennen, vielleicht gibt es dafür auch keine Namen. Gibt es etwas das so wahrscheinlich ist dass es nichts noch wahrscheinlicheres gibt?\n'+
    '\nGibt es das wahrscheinlichste Ereignis']+JN]
D+= [[u'Wie sieht es aus in der anderen Richtung? Kann man immer etwas finden das noch weniger und noch weniger wahrscheinlicher ist?'+
    ' Kann man das so immer unendlich weiter machen oder gibt es da ein Ende?\n'+
    '\nHilfestellung: Du musst die Sachen nicht nennen, vielleicht gibt es dafür auch keine Namen. Gibt es etwas das so wenig wahrscheinlich ist dass es nichts noch weniger wahrscheinlicheres gibt?\n'+
    '\nGibt es das unwahrscheinlichste Ereignis']+JN] 
D+= [[u'Kann es etwas geben das an sich nicht sicher ist und das wahrscheinlicher ist als eine sichere Sache?'+
    '\nHilfestellung: Mit weißen leeren Kärtchen die sicheren Ereignisse darstellen.']+JN] 
D+= [[u'Kann es etwas geben das an sich nicht unmöglich ist und das weniger wahrscheinlich ist als ein unmögliche Sache?'+
    '\nHilfestellung: Mit weißen leeren Kärtchen die unmöglichen Ereignisse darstellen.']+JN] 
D+= [[u'Hier siehst du eine Linie mit zehn Plätzen. In dieser Richtung ist es mehr wahrscheinlich und '+
    u'in dieser Richtung ist es weniger wahrscheinlich. Deine Aufgabe ist die Beispiele den Plätzen zuordnen. '+
    u'Wenn du glaubst zwei Sachen kommen auf denselben Platz, lege diese einfach so nebeneinander. '+
    u'Du darst die Reihenfolge jederzeit ändern.\n\n[Foto machen nachdem Kind fertig]''']+L]
D+= [[u'Das heißt, du würdest sagen dass das [Beispiel vorlesen] wahrscheinlicher ist als [Beispiel vorlesen].\n'+
    '[Alle Nachbarn so durchgehen]\n[Foto machen falls Reihenfolge geändert]\n'+
    'Ist es immer so dass etwas das höher liegt wascheinlicher ist als die Sachen darunter?']+JN]
    
D+= [[u'Hier siehst du eine lange Linie. In dieser Richtung ist es mehr wahrscheinlich und '+
    u'in dieser Richtung ist es weniger wahrscheinlich. Deine Aufgabe ist die Beispiele auf der Linie anzuordnen und zwar so.'+
    u'Wenn du glaubst zwei Sachen kommen auf dieselbe Stelle, lege diese einfach so untereinander.\n'+
    u'[Einverständniskopie für Eltern ausfüllen.]\nIst es immer so dass etwas das auf dieser Seite liegt wascheinlicher ist als die Sachen in dieser Richtung?''']+JN]
D+= [[u'Ende']+L]
#init stuff
resp=[]
SCALE=0.9
win=visual.Window(fullscr=True,size=(1900,600), units='norm',#pos=(0,0),
    color=(-1,0,1),winType='pyglet',screen=2)
instr=visual.TextStim(win,text='',height=0.1*SCALE,color='white',wrapWidth=2*SCALE,alignHoriz='center')
instr.setAutoDraw(True)
respTxt=visual.TextStim(win,text='',height=0.1*SCALE,pos=(0,-0.8*SCALE),wrapWidth=1*SCALE)
respTxt.setAutoDraw(True)  
bLeft=visual.TextStim(win,text=u'←',pos=(-0.9*SCALE,-0.9*SCALE),height=0.2*SCALE)
bRight=visual.TextStim(win,text=u'→',pos=(0.9*SCALE,-0.9*SCALE),height=0.2*SCALE)
bLeft.setAutoDraw(True); bRight.setAutoDraw(True)
# start  

states=[0]
resps=[]
while states[-1]<len(D):
    if not states[-1]: blc='grey'
    else: blc='white'
    if states[-1]==max(states): 
        brc='grey'
    else: brc='white'
    bLeft.setColor(blc)
    bRight.setColor(brc)
    respTxt.setColor('white')
    respTxt.setText(D[states[-1]][1])
    instr.setText(D[states[-1]][0])
    win.flip()
    #check response
    key=waitForKey(D[states[-1]][2]+['left','right','escape'])
    
    if key=='left':
        if blc=='white': states.append(states[-1]-1)
    elif key=='right': 
        if brc=='white': states.append(states[-1]+1)
    elif key=='escape': break
    else:
        D[states[-1]][3]=D[states[-1]][2].index(key)
        print(states[-1],D[states[-1]][3])
        states.append(states[-1]+1)
# derandomize eq classes
fr=open('vp%d.res'%vpid,'w')  
fr.write('%d,'%VERSION)
fl=open('vp%d.log'%vpid,'w') 
T=[]
for d in D:
    if len(d[2])>1: T.append(d[3])
    temp=''
    for g in d:
        tm=str(g)[:50].replace('\n',' ').replace('\t',' ')
        temp+=tm+u';'
    temp=temp[:-1]+u'\n' 
    fl.write(temp)
for ti in range(len(T)):
    if ti<len(W): fr.write(u'%d, '%T[rCIperm[ti]])
    else: fr.write(u'%d, '%T[ti])
for s in states:
    fl.write(str(s)+u',')
fl.write(str(rCIperm.tolist()))
ln=9*' ,'+' \n'
ln2='\'\',\'\'\n'
fr.write('\nm\nu\ns\nw\nu\n'+3*ln+6*ln2+',,,\ncomment')
fr.close()
fl.close() 
