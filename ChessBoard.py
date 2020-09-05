import turtle
t=turtle.Turtle()
t.speed('fastest')
i,j,z=0,7,1

while j>0:
    while i<4:
        for row in range(j):
            t.begin_fill()
            for y in range(4):
                t.forward(20)
                t.left(90)
            t.end_fill()
            z+=1
            t.forward(20)
            t.color('black','black')if z%2==1 else t.color('black','white')
        t.forward(20)
        t.left(90)
        i+=1
    t.forward(20)
    t.left(90)
    t.forward(20)
    t.right(90)
    j-=2
    i=0

