# PROBLEM 1 
#
#
# Introduction------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Say "Hello, World!" With Python

print("Hello, World!")


# Python If-Else

#!/bin/python3

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int( input().strip() )
    
    if n<=100 and n>=1:
        if n % 2 == 1 or (n % 2 == 0 and n in range( 6, 21 ) ):
            print( 'Weird' )
        elif n % 2 == 0 and ( n in range( 2, 5 ) or n > 20 ) :
            print( 'Not Weird' )
    else:
        print( 'Constraints on input: 1<=n<=100 ')


# Arithmetic Operators

if __name__ == '__main__':
    a = int( input() )
    b = int( input() )
    print( a+b )
    print( a-b )
    print( a*b )


# Python: Division

if __name__ == '__main__':
    a = int(input() )
    b = int(input() )
    print( a//b )
    print( a/b )


# Loops

if __name__ == '__main__':
    n = int(input() )
    for i in range( 0,n ):
        print( i*i )


# Write a function

def is_leap(year):
    leap = False
    if year % 4 == 0:
        leap = True
        if year % 100 == 0 and year % 400 != 0:
            leap = False
    return leap


# Print Function

if __name__ == '__main__':
    n = int(input())
    my_str = ""
    for i in range(1,n+1):
        my_str = my_str + str(i)
    print(my_str)


# Data types -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    my_list=[[i,j,k] for i in range(0,x+1)
    for j in range(0,y+1)
    for k in range(0,z+1)
    if i+j+k != n]
    
    print(my_list)


# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    highest = max(arr)
    runner_up = max(i for i in arr if i!=highest)
    print(runner_up)

  
# Nested Lists

if __name__ == '__main__':
    
    my_list=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        my_list.append([name,score])
    
    lowest_score = min( i[1] for i in my_list )
    second_ls =  min([i[1] for i in my_list if i[1] != lowest_score] )
    names=sorted(i[0] for i in my_list if i[1] == second_ls)
    for i in names:
        print(i)


# Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    avg=round(sum(student_marks[query_name])/
    len(student_marks[query_name]),2)
    print(f"{avg:.2f}")



# Lists

if __name__ == '__main__':
    my_list=[]
    N = int(input())
    for i in range(N):
        x=input().split()

        if x[0]=='insert':
            my_list.insert(int(x[1]),int(x[2]))
        elif x[0]=='print':
            print(my_list)
        elif x[0]=='remove':
            my_list.remove(int(x[1]))
        elif x[0]=='append':
            my_list.append(int(x[1]))
        elif x[0]=='sort':
            my_list.sort()
        elif x[0]=='reverse':
            my_list.reverse()
        elif x[0]=='pop':
            my_list.pop()


# Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(int(i) for i in integer_list)
    print(hash(t))


# Strings -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# sWAP cASE

def swap_case(s):
    x = ''
    for i in s:
        if i.isupper() == 1:
            i = i.lower()
        else:
            i = i.upper()
        x = x+i
    return x
  

# String Split and Join

def split_and_join(line):
    s=line.split()
    s='-'.join(s)
    return s


# What's Your Name?

def print_full_name(first, last):
    x = f"Hello {first} {last}! You just delved into python."
    print(x)


# Mutations

def mutate_string(string, position, character):
    string=string[:position]+character+string[position+1:]
    return string


# Find a string

def count_substring(string, sub_string):
    #not converting strings to upper case because of case-sensitivity
    
    #initializing counter
    c = 0
    #for loop: if starting at index i, the sliced string from i to i+length of substring
    #is equal to the substring, then we increment the counter.
    #Allows to check overlapping occurences

    for i in range(0,len(string)-len(sub_string)+1):
      if string[i:i+len(sub_string)] == sub_string:
        c+=1
    return c


# String Validators

if __name__ == '__main__':
    s = input()
    a=0
    b=0
    c=0
    d=0
    e=0
    for i in s:
        a+= i.isalnum()
        b+= i.isalpha()
        c+= i.isdigit()
        d+= i.islower()
        e+= i.isupper()
    print( a>0 )
    print( b>0 )
    print( c>0 )
    print( d>0 ) 
    print( e>0 )


# Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# Text Wrap

def wrap(string, max_width):
   slices=[]
   #empty list to store the subarrays of length max_width
   
   for i in range(0,len(string),max_width):
       slices.append(string[i:i+max_width])

  #join them with the new line char
   my_string='\n'.join(slices)
   return my_string


# Designer Door Mat

N,M = input().split()
c='.|.'
#upper part
for i in range(int(N)//2):
    pattern=c*(2*i+1)
    print(pattern.center(int(M),'-'))
#middle part
print('WELCOME'.center(int(M),'-'))
#bottom
for i in range(int(N)//2-1,-1,-1):
    pattern=c*(2*i+1)
    print(pattern.center(int(M),'-'))


# String Formatting

def print_formatted(number):
    # your code goes here
    width = len(format(number, 'b'))
    for i in range(1,number+1):
      print(f"{i:{width}d} {format(i,'o').rjust(width)} {format(i,'X').rjust(width)} {format(i,'b').rjust(width)}")


# Alphabet Rangoli

import string

def print_rangoli(size):
    #alphabet
    alpha = string.ascii_lowercase
    my_list = []
    
    #pattern
    for i in range(size):
        # Slice the string and reverse
        s = '-'.join(alpha[size-1:i:-1] + alpha[i:size])
        # Center the string in the list
        my_list.append(s.center(4*size-3, '-'))
    
    # Join the top half (with the middle) and the bottom half to form the complete rangoli
    print( '\n'.join(my_list[::-1] + my_list[1:]))


# Capitalize!

def solve(s):
    l = s[0].upper()+s[1:]
    for i in range(len(s)):
        if s[i-1] == ' ':
            l = l[:i]+l[i].upper()+l[i+1:]
    return l


# The Minion Game

def minion_game(string):
    score_s = 0
    score_k = 0
    vowels = 'AEIOU'
    
    for i in range(len(s)):
        if s[i] in vowels:
            #all possible subs starting at i are lens-i
            score_k+=len(s)-i
        else:
            score_s+=len(s)-i    
    
    if score_s > score_k:
        out = f'Stuart {score_s}'
    elif score_s < score_k:
        out = f'Kevin {score_k}'
    else:
        out = 'Draw'
    print(out)


# Merge the Tools!

def merge_the_tools(string, k):
    n=len(string)
    for i in range(0,n,k):
        t_i=string[i:i+k]
        u_i = ''
        seen = set()
        for j in t_i:
            if j not in seen:
                u_i+=j
                seen.add(j)
        print(u_i)


# Sets-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Introduction to Sets

def average(array):
    heights = set(array)
    hsum = 0
    for i in heights:
        hsum+=i
    avh = round(hsum/len(heights),3)
    return avh


# No Idea!

import array as r
n,m = map( int, ( input().split() ))
arr=r.array('i', map(int, input().split() ))
A=set(map( int, input().split() ))
B=set(map( int, input().split() ))
happiness=0
for i in arr:
    if i in A:
        happiness+=1
    elif i in B:
        happiness-=1
print(happiness)


# Symmetric Difference

M=int(input())
listM=list(map(int,input().split() ))
a=set(listM)
N=int(input())
listN=list(map(int,input().split() ))
b=set(listN)

simm_diff = sorted((a.difference(b)).union(b.difference(a)))

for i in simm_diff:
  print(i)


# Set .add()

N=int(input())
countries=set()
for i in range(N):
    country=input()
    countries.add(country)
print(len(countries))


# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
N=int(input())
for i in range(N):
    command = input().split()
    if command[0] == 'pop':
        s.pop()
    elif command[0] == 'remove':
        s.remove(int(command[1]))
    elif command[0] == 'discard':
        s.discard(int(command[1]))
sums = sum(i for i in s)
print(sums)


# Set .union() Operation

n = int(input())
n_news = set(map(int,input().split()))
b = int(input())
b_news = set(map(int,input().split()))
out = len(n_news.union(b_news))
print(out)


# Set .intersection() Operation

n = int(input())
n_news = set(map(int,input().split()))
b = int(input())
b_news = set(map(int,input().split()))
out = len(b_news.intersection(n_news))
print(out)


# Set .difference() Operation

n = int(input())
n_news = set(map(int,input().split()))
b = int(input())
b_news = set(map(int,input().split()))
eng_only = len(n_news.difference(b_news))
print(eng_only)


# Set .symmetric_difference() Operation

n = int(input())
n_news = set(map(int,input().split()))
b = int(input())
b_news = set(map(int,input().split()))
notboth = len(n_news.symmetric_difference(b_news))
print(notboth)


# Set Mutations

n = int(input())
A = set(map(int,input().split()))
N = int(input())
for i in range(N):
    operation=input().split()
    set_i=set(map(int,input().split()))
    if operation[0] == 'update':
        A.update(set_i)
    if operation[0] == 'difference_update':
        A.difference_update(set_i)
    if operation[0] == 'intersection_update':
        A.intersection_update(set_i)
    if operation[0] == 'symmetric_difference_update':
        A.symmetric_difference_update(set_i)
print(sum(A))


# The Captain's Room

K = int(input())                           #size of each group
room_no = list(map(int,input().split()))   #list of room numbers
sum_all = sum(room_no)
uniques = set(room_no)
sum_uniques = sum(uniques)
captain_room = (K*sum_uniques-sum_all)//(K-1)    
print(captain_room)


# Check Subset

T = int(input())
for i in range(T):
    card_A = int(input())
    A = set(map(int,input().split()))
    card_B = int(input())
    B = set(map(int,input().split()))
    print(A.difference(B) == set())         #A is a subset of B if the set difference A\B is the empty set

  
# Check Strict Superset

A = set(map(int,input().split()))
n = int(input())
count = 0
for i in range(n):
    N_i = set(map(int,input().split()))
    if N_i.difference(A) == set() and len(N_i) < len(A):
        #N_i is a proprt subset of A if the set difference N_i\A is empty
        #and |N_i|<|A| (A must have at least one mor element)
        count+=1
print(count == n)                               # A is a strict super set of all n sets if the condition is the loop is true for every N_i which means that count=n


# CCollections ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# collections.Counter()

from collections import Counter
X = int(input())
sizes = list(map(int, input().split()))
av_sizes = Counter(sizes)
N = int(input())                    #costumers no.
income = 0
for i in range(N):
    costumer_i = input().split()
    size_i = int(costumer_i[0])
    x_i = int(costumer_i[1])
    if av_sizes[size_i] > 0:        #if there's still the number 
        income += x_i               #add the price to income
        av_sizes[size_i] -= 1       #and decrement the no. of that size available
print(income)


# DefaultDict Tutorial

from collections import defaultdict

n,m = map(int,input().split())
A = defaultdict(list) 
B = []
for i in range(n):
    n_i = input()
    A[n_i].append(i+1)
for j in range(m):
    m_j = input()
    B.append(m_j)
for l in B:
    if l in A:
        print(' '.join(map(str,A[l])))       #to format the values of the key l as a space separated string of numbers                             
    else:
        print(-1)


# Collections.namedtuple()

from collections import namedtuple
N, columns = int(input()), input().split()
Student = namedtuple('Student', columns)
print(round(sum ([int(Student(*input().split()).MARKS) for i in range(N)]) / N , 2 ) )


# Collections.OrderedDict()

from collections import OrderedDict

N = int(input())
items = OrderedDict()
for i in range(N):
    *item_i, price_i = input().rsplit(' ',1)
    item_i =' '.join(item_i)
    price_i = int(price_i)
    if item_i in items:
        items[item_i] += price_i
    else:
        items[item_i] = price_i
for i,j in items.items():
    print(i, j)


# Word Order

from collections import OrderedDict
n=int(input())
words=OrderedDict()
for i in range(n):
    word=input()
    if word in words:
        words[word]+=1
    else:
        words[word]=words.setdefault(word,0)+1
print(f"{len(words)}\n{' '.join(map(str, words.values() ))}" )


# Collections.deque()

from collections import deque
N = int(input())
d = deque()
for i in range(N):
    command=list(input().split())
    if command[0]=='append':
        d.append(int(command[1]))
    elif command[0]=='appendleft':
        d.appendleft(int(command[1]))
    elif command[0]=='clear':
        d.clear()
    elif command[0]=='count':
        d.count(command[1])
    elif command[0]=='pop':
        d.pop()
    elif command[0]=='popleft':
        d.popleft()
    elif command[0]=='extend':
        d.extend(command[1])
    elif command[0]=='extendleft':
        d.extendleft(command[1])
    elif command[0]=='remove':
        d.remove(command[1])
    elif command[0]=='reverse':
        d.reverse()
    elif command[0]=='rotate':
        d.rotate(int(command[1]))
print(' '.join(map(str,d)))
        

# Company Logo
#!/bin/python3

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    s = input()
    d=dict()
    
    for i in s:
        if i in d:
            d[i]+=1
        else:
             d[i]=1
      
    d=sorted(d.items(), key=lambda item: ( -item[1], item[0] ) )        #sorting in descending order based on the values first, then alphabetically based on the keys
    
    for key, value in d[:3]:
        print(f"{key} {value}")


# Piling Up!

from collections import deque
T = int(input())
b = deque()

for i in range(T):
    n = int(input())
    blocks = deque(map(int, input().split()))
    possible = True
    last = float('inf')  #initializing last on pile to be inf so that the first block picked is always smaller 
    
    while blocks:
        
        if blocks[0]>=blocks[-1]:
            pick=blocks.popleft()
            
        else:
            pick=blocks.pop()
        #update to not possible if current is bigger then last on vertical pile
        if pick>last:
            possible=False
            break
        
        #the picked one is now the last on the pile
        last=pick
            
    if possible:
        print('Yes')
    else:
        print('No')


# Date and Time ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calendar Module

import calendar

MM,DD,YYYY=list(map(int, input().split()))
week = ['Monday','Tuesday','Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday']
day = calendar.weekday(YYYY,MM,DD)
print(week[day].upper())


# Time Delta
#!/bin/python3

import math
import os
import random
import re
import sys
import calendar
from datetime import datetime

def time_delta(t1, t2):
    # Convert datetime objects with timezone info
    dt1 = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    dt2 = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    diff = abs((dt1 - dt2).total_seconds())
    return int(diff)
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
  
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(str(delta) + '\n')

  fptr.close()

        
# Exceptions ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Exceptions

T = int(input())
for i in range(T):
    try:
        a,b = map(int, input().split())
        print(a//b)
    except ZeroDivisionError:
        print("Error Code: integer division or modulo by zero")
    except ValueError as e:
        print(f"Error Code: {e}")


#  Built-ins ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Zipped!

N, X = map(int, input().split())
grades = []

for i in range(X):
    grades.append(list(map(float, input().split())))
    
for student in zip(*grades):
    print( round( sum(student)/X, 1 ) )
  
  
# Athlete Sort
#!/bin/python3

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
    arr.sort(key=lambda x : x[k])
    
    for row in arr:
        print(' '.join(map(str, row)))


# ginortS

s = input()
out = sorted(s, key=lambda x:(x.isdigit(), x.isdigit() and int(x)%2==0, x.isupper(),x)) #the last x helps to sort lexicographically. x.isdigit pushes the digits after
print(''.join(out))


# Python Functionals --------------------------------------------------------------------------------------------------------------------------------------------------------------

# Map and Lambda Function

cube = lambda x: x**3 

def fibonacci(n):
    
    if n == 0:
        return []
    elif n == 1:
        return [0]
        
    #initialized afterwords, or else you'll get [0,1] even if n=0 (seen from last wrong submission)
    
    fib = [0,1]
    for i in range(2,n):
        fib.append( fib[i-1] + fib[i-2] )
    return fib


# Regex and Parsing challenges-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Detect Floating Point Number

import re

pattern = r'^[+-]?\d*\.\d+$'  #starts with + or -,  . must be there, allows .4, at list one decimal

T=int(input())
for i in range(T):
    N=input()
    out=False                 #initialize
    
                              #re.matches to check if the first input is in the second
    if re.match(pattern, N):
                              #check if it's possible to convert to float
        try:
            float(N)
            out=True
        except ValueError:
            out=False
        
    print(out)


# Re.split()

regex_pattern = r"[,.]"	# it will then split when , or . occurs

import re
print("\n".join(re.split(regex_pattern, input())))


# Group(), Groups() & Groupdict()

import re
s = input()
alnum = re.search(r'([a-zA-Z0-9])\1+', s)
if alnum:
    print(alnum.group(1))
else:
    print(-1)


# Re.findall() & Re.finditer()

import re
s = input()
#consonant left (lookbehind:?<=), 2 or more vowels substring {2,}, consonant right (look haed ?=) 
pattern = r'(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])[aeiouAEIOU]{2,}(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])'
subs = re.findall(pattern,s)
if subs:
    for i in subs:
        print(i)
else:
    print(-1)


# Re.start() & Re.end()

import re
s=input()
k=input()
m = list(re.finditer(f'(?={k})', s))  #allows overlaps
if m:
    for i in m:
        print((i.start(),i.start()+len(k)-1))
else:
    print((-1,-1))


# Regex Substitution

import re

N=int(input())
for i in range(N):
    line=input()
    out = re.sub(r'(?<= )&&(?= )', 'and', line)
    out = re.sub(r'(?<= )\|\|(?= )', 'or', out)
    print(out)


# Validating Roman Numerals

regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"	
import re
print(str(bool(re.match(regex_pattern, input()))))


# Validating phone numbers

import re
regex_pattern = r"^[789]\d{9}$"    # 10-digits starting with 7, 8, or 9
N = int(input())                   # Number of mobile numbers
for i in range(N):
    number = input().strip()   
    # Check if the number matches the regex pattern
    if bool(re.match(regex_pattern, number)):
        print('YES')
    else:
        print('NO')


# Validating and Parsing Email Addresses

import re
import email.utils

regex_pattern = r'^[a-zA-Z][\w\.-]+@[a-zA-Z]+\.[a-zA-Z]{1,3}$'

n = int(input()) 

for i in range(n):
    x = input().strip()
    name, email_add= email.utils.parseaddr(x)  # Parse name and email
    
    # Check if email matches pattern
    if re.match(regex_pattern, email_add):
        print(email.utils.formataddr((name, email_add)))



# Hex Color Code

import re
N = int(input())       # number of lines
codes = []
inside_braces = False  # A app var to check CSS properties
                       # Read the code lines
for i in range(N):
    line = input()
    codes.append(line)
                       # Regex pattern to match hex color codes
col_pattern = r'#[0-9A-Fa-f]{3,6}'
colors = []
for line in codes:
                       # Split the line by '{' and '}' to handle multiple braces in a single line
    parts = re.split(r'([{}])', line)
    for part in parts:
        if part == '{':
            inside_braces = True
        elif part == '}':
            inside_braces = False
        elif inside_braces:
                       # Find all HEX color codes in the current segment
            matches = re.findall(col_pattern, part)
            colors.extend(matches)
                       # Print the color codes line by line
for color in colors:
    print(color)


# HTML Parser - Part 1

from html.parser import HTMLParser
# subclass and override the methods
class MyHTMLParser(HTMLParser):
    
    def handle_starttag(self, tag, attrs):
        print (f"Start : {tag}")
        for i in attrs:
            # Print attribute name and value
            print(f"-> {i[0]} > {i[1] if i[1] else 'None'}")
            
    def handle_endtag(self, tag):
        print (f"End   : {tag}")
        
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for i in attrs:
            print(f"-> {i[0]} > {i[1] if i[1] else 'None'}")

N = int(input()) #lines number to loop over
html_input =''.join(input().strip() for i in range(N))  # Read lines   
parser = MyHTMLParser()
parser.feed(html_input)


# HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    
    def handle_comment(self, data):
        if "\n" in data: #has multiple lines of commen
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data)
    
    def handle_data(self, data):
        if data != '\n': #constraint: no empty data
            print (">>> Data")
            print(data)
  
html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    
    def handle_starttag(self, tag, attrs):
        print(tag)
        # If there are attributes, print them
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

    def handle_endtag(self, tag):
        pass  # We don't need to do anything for end tags

    def handle_startendtag(self, tag, attrs):
        # Handle self-closing tags
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

    def handle_comment(self, data):
        pass  # Ignore comments

    def handle_data(self, data):
        pass  # Ignore data outside of tags

n = int(input().strip())  # Read number of lines
html_input = ''.join(input().strip() for _ in range(n))  # Read HTML lines
parser = MyHTMLParser()
parser.feed(html_input)  # Feed the HTML to the parser


# Validating UID

import re

pattern = r'^(?=(?:[^A-Z]*[A-Z]){2})(?=(?:[^0-9]*[0-9]){3})(?!.*(.).*\1)[A-Za-z0-9]{10}$'
n = int(input()) #testcases
for i in range(n):
    uid = input()   #read uid
    if re.match(pattern, uid):  #match with pattern
        print("Valid")
    else:
        print ("Invalid")


# Validating Credit Card Numbers

import re
#?!.*(\d)(?:-?\1){3} negative lookhaed:
#max rep of same digit after one is 3
#[4-6]\d{3} first is 4,5 or 6 followed by 3 digits (sequences of 4)
#(-?\d{4}){3}:3 groups of 4, optional -
pattern = r'^(?!.*(\d)(?:-?\1){3})([4-6]\d{3}(-?\d{4}){3})$'
n = int(input())
for i in range(n):
    card_no = input()
    if re.match(pattern, card_no):
        print('Valid')
    else:
        print('Invalid')


# Validating Postal Codes

regex_integer_in_range = r'^[1-9][0-9]{5}$'	                     # the first cannot be 0 (starts with:^), the next 5 from 0 to 9.
regex_alternating_repetitive_digit_pair = r'(?=(\d)(?=\d\1))'    #(?=...):  positive lookahead to check for a condition without consuming characters.
#(?=\d\1): the next character is a digit followed by the same digit captured in the first group. ensures there's exactly one digit in between the two repeating digits.

import re
P = input()
print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)


# Matrix Script
#!/bin/python3

import math
import os
import random
import re
import sys

#  non-alphanumeric chars
pattern = r'(?<=\w)([^\w]+)(?=\w)'
first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
  
#translate matrix in a string by joining columns and rows elements
string = ''.join(matrix[i][j] for j in range(m) for i in range(n))
#repleces pattern with tab
new_matrix= re.sub(pattern, ' ', string)
print(new_matrix)


# XML --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# XML 1 - Find the Score

def get_attr_number(node):
    # Initialize
    score = len(node.attrib)
    
    # using recursion to add score of child nodes
    for child in node:
        score += get_attr_number(child)
    
    return score


# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    #add level
    level += 1
    # Update maxdepth if the current level is greater than maxdepth
    if level > maxdepth:
        maxdepth = level
    # Recursively call depth for all children of the current element
    for child in elem:
        depth(child, level)


# Closures and Decorations ------------------------------------------------------------------------------------------------------------------------------------------------------------

# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        #slicing input to have xxxxx xxxxx
        standard_format = ['+91'+' '+number[-10:-5]+' '+number[-5:]for number in l]
        return f(standard_format)
    return fun


# Decorators 2 - Name Directory

from operator import itemgetter

def person_lister(f):
    def inner(people):
        return [f(p) for p in sorted(people,key = lambda x: int(x[-2]))]
    return inner

# Numpy---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Arrays

def arrays(arr):
   return numpy.array( arr, float )[::-1]
  

# Shape and Reshape

import numpy

l = list( map( int, input().split() ) )
arr = numpy.array(l)
arr.shape = (3,3)
print (arr)

# Transpose and Flatten

import numpy

N, M = map( int, input().split() )
matrix = numpy.zeros((N,M), dtype=int)

for i in range(N):
    row = numpy.array( list(map(int, input().split())))
    matrix[i] = row

print( numpy.transpose( matrix ))
print( matrix.flatten() )


# Concatenate

import numpy as np

N,M,P = map(int, input().split())
array_1 = np.empty((N,P), dtype = int)
array_2 = np.empty((M,P), dtype = int)

for i in range(N):
    row_i = list(map(int, input().split()))
    array_1[i] = row_i

for i in range(M):
    row_i = list(map(int, input().split() ))
    array_2[i] = row_i
    
print(np.concatenate((array_1,array_2), axis=0))


# Zeros and Ones

import numpy as np

dims = list(map(int, input().split()))

print(np.zeros(dims, dtype=int))
print(np.ones(dims, dtype=int))


# Eye and Identity

import numpy as np
np.set_printoptions(legacy='1.13')

N,M = map(int, input().split())
print(np.eye(N,M))


# Array Mathematics

import numpy as np

N, M = map(int, input().split())
A = np.empty((N,M), dtype=int)
B = np.empty((N,M), dtype=int)

for i in range(N):
    A[i] = list(map(int, input().split()))
for i in range(N):
    B[i] = list(map(int, input().split()))

print(A+B)
print(A-B)
print(A*B) 
print(A//B)
print(A % B)
print(A**B)


# Floor, Ceil and Rint

import numpy as np
np.set_printoptions(legacy='1.13')

A = list( map(float, input().split() ))

A = np.array(A)

print(np.floor(A))
print(np.ceil(A))
print(np.rint(A))


# Sum and Prod

import numpy as np

N,M = map(int, input().split())
arr = np.empty((N,M), dtype = int)
for i in range(N):
    arr[i] = list(map(int, input().split() ))

print( np.prod( np.sum(arr, axis = 0), axis = None))


# Min and Max

import numpy as np

N,M = map(int, input().split())
arr = np.empty((N,M), dtype = int)
for i in range(N):
    arr[i] = list(map(int, input().split()))
print(np.max((np.min(arr, axis = 1))))


# Mean, Var, and Std

import numpy as np

N,M = map(int, input().split())
arr = np.empty((N,M), dtype=int)

for i in range(N):
    arr[i] = list(map(int, input().split()))
    
print(np.mean(arr, axis = 1))
print(np.var(arr, axis = 0))
print(round(np.std(arr, axis = None), 11)) #rounding up to 11 decimals because of py version


# Dot and Cross

import numpy as np
N = int(input())
A = np.empty((N,N), dtype = int)
B = np.empty((N,N), dtype = int)

for i in range(N):
    A[i] = list(map(int, input().split()))
for i in range(N):
    B[i] = list(map(int, input().split()))

print(np.dot(A,B))


# Inner and Outer

import numpy as np
A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))

print(np.inner(A,B))
print(np.outer(A,B))


# Polynomials

import numpy as np

P_coeff = list(map(float, input().split()))
x = float(input())
print(np.polyval(P_coeff,x))


# Linear Algebra

import numpy as np
N = int(input())
A = np.empty((N,N))

for i in range(N):
    A[i] = list(map(float, input().split()))

print(round(np.linalg.det(A),2))


#
#
# PROBLEM 2
#
#


# Birthday Cake Candles
#!/bin/python3

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    
    m = max(candles) #find the max
    return candles.count(m) #count the occurrencies of it

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()



# Number Line Jumps

#!/bin/python3

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    
    if v1 <= v2:                     #first kang will never catch up because x1<x2 is always true
        out = 'NO'
    else:
                                     #check if the difference in starting positions is divisible by the difference in jumps
        if (x2-x1) % (v1-v2) == 0:   #well defined (v1!=v2): 
            out = 'YES'
        else:
            out = 'NO'
            
    return out

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()



# Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    
    shared = 5       #initialize
    tot_likes = 0
    
    for i in range(n):
        liked = math.floor( shared / 2 )
        shared = liked*3
        tot_likes += liked
        
    return tot_likes

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()



# Recursive Digit Sum

#!/bin/python3

import math
import os
import random
import re
import sys


def superDigit(n, k):
    sum_0 = sum( int(i) for i in n ) * k
    
    def recursive(p):
        if len(p) == 1:
            return int(p)
        # Otherwise, sum the digits and recursively call the function 
        # converting p to string
        else:
            p_sum = sum(int(i) for i in p)
        return recursive(str(p_sum))
    
        # Call recursive function to compute the super digit it
    return recursive(str(sum_0))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()



# Insertion Sort - Part 1

#!/bin/python3

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    x = arr[n-1]
    i = n-1
    while i > 0 and x < arr[i-1]:
        arr[i] = arr[i-1]                    # Shift the element to the right
        print(' '.join(map(str,arr)))
        i -= 1
        
    # Insert the element into its correct position
    arr[i] = x
    print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# Insertion Sort - Part 2

#!/bin/python3

import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    #loop sort1 function but left to right
    for i in range(1,n):
        x = arr[i]
        j = i-1
        while j >= 0 and arr[ j ] > x:
            arr[ j+1 ] = arr[ j ]  # Shift the element to the right
            j -= 1

        arr[j+1] = x   # Insert the element into its correct position
        
        print(' '.join(map(str, arr)))
    

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
