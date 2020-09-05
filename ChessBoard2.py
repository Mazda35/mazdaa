from turtle import *
i,z=[8,7,7,6,6,5,5,4,4,3,3,2,2,1],1
t=Turtle()
t.speed('fastest')
for x in i:
    for row in range(x):
        t.begin_fill()
        for y in range(4):
            t.forward(20)
            t.left(90)
        t.end_fill()
        z += 1
        t.forward(20)
        t.color('black', 'black') if z % 2 == 1 else t.color('black', 'white')
    t.left(90)
    t.forward(20)
t.left(90)
t.forward(20)








