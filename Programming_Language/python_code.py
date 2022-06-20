with open('input.txt', 'r') as f :
    a = f.readlines()

total = []
int_list = []
char_list = []
add_count = 0
for x in range(0, len(a)) :
    b = list(a[x])
    c = []
    count = 0
    for y in range(0, len(b)) :
        if count <= 0 :
            if b[y] == '+' :
                add_count = add_count + 1
            if b[y] == 'i' and b[y + 1] == 'n':
                c.append('int')
                count = 3
                int_list.append(b[y + 4])
            elif b[y] == 'c' and b[y + 1] == 'h':
                c.append('char')
                count = 4
                char_list.append(b[y + 5])
            elif b[y] == 'p' and b[y + 1] == 'r' :
                c.append('project')
                count = 7
            elif b[y] == 'e' and b[y + 1] == 'x':
                c.append('exit')
                count = 4
            elif b[y] == ' ' or b[y] == '\n':
                count = 0
                continue
            else :
                c.append(b[y])
                count = 0
        count = count - 1
    total.append(c)

total_stack0 = []
print('----- ----- ----- ----- ----- Scanner ------ ----- ----- ----- -----')
for x in range(len(total)) :
    for y in range(len(total[x])) :
        print('token :', total[x][y])
        total_stack0.append(total[x][y])

total_stack0.append('$')
total_stack = []
for x in range (len(total_stack0)) :
    top = total_stack0.pop()
    total_stack.append(top)
stack = []
stack.append('$0')

print()
print('----- ----- ----- ----- ----- Parser ----- ----- ----- ----- -----')
p0 = []
p1 = []
p2 = []
p3 = []
p4 = []
p5 = []
p6 = []
p7 = []
p8 = []
p9 = []
p10 = []
p11 = []
p12 = []
p13 = []
p14 = []
p15 = []
p16 = []
p17 = []
p18 = []
p19 = []
p20 = []
p21 = []
p22 = []
p23 = []
p24 = []
p25 = []
p26 = []
p27 = []
p28 = []
p29 = []
p30 = []
p31 = []
p32 = []
p33 = []
p34 = []
p35 = []
p36 = []
p37 = []
p38 = []
p39 = []
p40 = []
p41 = []
p42 = []
p43 = []
p44 = []
p45 = []
p46 = []
p47 = []

print(total_stack)
print(stack)
count = 0
while True :
    if count == 0 :
        top = total_stack.pop()
        num = 2
        temp = top + str(num)
        stack.append(temp)

    print(total_stack)
    print(stack)

    count = count + 1
    if count == 1 :
        break

num_reg = 0
flist = []
print()
print('----- ----- ----- ----- ----- Code Generator ----- ----- ----- ----- -----')
print('Begin  project')
for x in range(len(int_list)) :
    flist.append('reg' + str(x + 1))
    numb = 'reg' + str(x + 1) + ','
    print('LD    ', numb, int_list[x])
for x in range(len(char_list)):
    numb = 'reg' + str(x + 1 + len(int_list)) + ','
    print('LD    ', numb, char_list[x])
for x in range(add_count):
    fin = flist[len(flist) - 1] + ', '
    fin = fin + flist[len(flist) - 3] + ', '
    fin = fin + flist[len(flist) - 2]
    print('ADD   ', fin)
print('ST    ', flist[len(flist) - 1])
print('End    project')
print()
print('Register # :', len(int_list))