# Algorithm 1: Calculate the sum of the first n natural numbers
a = 10
b = 0
c = 1
while c <= a:
	d = c
	b = b + d
	c = c + 1
print(b)

# Algorithm 2: Calculate the factorial of a number
a = 5
b = 1
c = 1
while c <= a:
	d = c
	b = b * d
	c = c + 1
print(b)

# Algorithm 3: Check if a number is prime
a = 29
b = 2
c = 1
while b < a:
	d = a % b
	if d == 0:
		c = 0
	b = b + 1
print(c)

# Algorithm 4: Calculate the greatest common divisor (GCD)
a = 48
b = 18
while b > 0:
	c = a % b
	a = b
	b = c
print(a)

# Algorithm 5: Reverse a number
a = 1234
b = 0
while a > 0:
	c = a % 10
	d = b * 10
	b = d + c
	e = a - c
	a = e / 10
print(b)

# Algorithm 6: Calculate the power of a number
a = 2
b = 5
c = 1
d = 1
while d <= b:
	c = c * a
	d = d + 1
print(c)

# Algorithm 7: Find the sum of digits of a number
a = 1234
b = 0
while a > 0:
	c = a % 10
	b = b + c
	a = a // 10
print(b)

# Algorithm 8: Check if a number is an Armstrong number
a = 153
b = a
c = 0
while b > 0:
	d = b % 10
	e = d * d
	f = e * d
	c = c + f
	b = b // 10
if c == a:
	print(1)
else:
	print(0)

# Algorithm 9: Generate Fibonacci sequence up to n terms
a = 10
b = 0
c = 1
d = 0
while d < a:
	print(b)
	e = b + c
	b = c
	c = e
	d = d + 1

# Algorithm 10: Count the number of digits in a number
a = 1234
b = 0
while a > 0:
	b = b + 1
	c = a % 10
	a = a // 10
print(b)

# Algorithm 11: Check if a number is a palindrome
a = 121
b = a
c = 0
while b > 0:
	d = b % 10
	e = c * 10
	c = e + d
	b = b // 10
if c == a:
	print(1)
else:
	print(0)

# Algorithm 12: Find the square root of a number using approximation
a = 25
b = a
c = 0.1
while b * b > a:
	b = b - c
print(b)

# Algorithm 13: Find the sum of the squares of the first n natural numbers
a = 5
b = 0
c = 1
while c <= a:
	d = c * c
	b = b + d
	c = c + 1
print(b)

# Algorithm 14: Find the product of the digits of a number
a = 123
b = 1
while a > 0:
	c = a % 10
	b = b * c
	a = a // 10
print(b)

# Algorithm 15: Find the largest digit in a number
a = 38924
b = 0
while a > 0:
	c = a % 10
	if c > b:
		b = c
	a = a // 10
print(b)

# Algorithm 16: Find the smallest digit in a number
a = 38924
b = 9
while a > 0:
	c = a % 10
	if c < b:
		b = c
	a = a // 10
print(b)

# Algorithm 17: Convert binary to decimal
a = 1011
b = 0
c = 0
while a > 0:
	d = a % 10
	e = 1
	p = c
	while p > 0:
		e = e * 2
		p = p - 1
	f = d * e
	b = b + f
	a = a // 10
	c = c + 1
print(b)

# Algorithm 18: Convert decimal to binary
a = 10
b = 0
c = 1
while a > 0:
	d = a % 2
	e = d * c
	b = b + e
	c = c * 10
	a = f // 2
print(b)

# Algorithm 19: Check if a number is a perfect square
a = 16
b = 1
c = 0
while b * b <= a:
	if b * b == a:
		c = 1
	b = b + 1
print(c)

# Algorithm 20: Check if a number is a perfect cube
a = 27
b = 1
c = 0
while b * b * b <= a:
	if b * b * b == a:
		c = 1
	b = b + 1
print(c)

# Algorithm 21: Find the sum of the cubes of the first n natural numbers
a = 5
b = 0
c = 1
while c <= a:
	d = c * c
	e = d * c
	b = b + e
	c = c + 1
print(b)

# Algorithm 22: Calculate the nth triangular number
a = 7
b = 0
c = 1
while c <= a:
	b = b + c
	c = c + 1
print(b)

# Algorithm 23: Count the number of trailing zeros in a factorial
a = 10
b = 0
c = 5
while a >= c:
	d = a / c
	e = d % 10
	e = d - e
	b = b + e
	c = c * 5
print(b)

# Algorithm 24: Check if a number is an automorphic number
a = 25
b = a * a
c = 1
while a > 0:
	d = a % 10
	e = b % 10
	if d != e:
		c = 0
	f = a - d
	g = b - e
	a = f / 10
	b = g / 10
print(c)

# Algorithm 25: Calculate the harmonic sum of the first n natural numbers
a = 5
b = 0
c = 1
while c <= a:
	d = 1 / c
	b = b + d
	c = c + 1
print(b)

# Algorithm 26: Calculate the nth Fibonacci number
a = 10
b = 0
c = 1
d = 2
while d <= a:
	e = b + c
	b = c
	c = e
	d = d + 1
print(c)

# Algorithm 27: Check if a number is a Harshad number
a = 18
b = a
c = 0
while b > 0:
	d = b % 10
	c = c + d
	e = b - d
	b = e / 10
f = a % c
if f == 0:
	print(1)
else:
	print(0)

# Algorithm 28: Find the LCM of two numbers
a = 12
b = 15
c = a
while c % b != 0:
	c = c + a
print(c)

# Algorithm 29: Check if a number is a palindrome (alternative)
a = 1221
b = 0
c = a
while c > 0:
	d = c % 10
	e = b * 10
	b = e + d
	f = c - d
	c = f / 10
if a == b:
	print(1)
else:
	print(0)

# Algorithm 30: Calculate the digital root of a number
a = 9875
while a > 9:
	b = 0
	while a > 0:
		c = a % 10
		b = b + c
		d = a - c
		a = d / 10
	a = b
print(a)