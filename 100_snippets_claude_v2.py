#0. description: Simple addition
# code
a = 5
b = 3
c = a + b
print(c)

#1. description: Simple subtraction
# code
a = 10
b = 4
c = a - b
print(c)

#2. description: Simple multiplication
# code
a = 6
b = 7
c = a * b
print(c)

#3. description: Simple division
# code
a = 15
b = 3
c = a / b
print(c)

#4. description: Find maximum of two numbers
# code
a = 10
b = 20
c = a
x = b
if x > c:
	c = x
print(c)

#5. description: Find minimum of two numbers
# code
a = 10
b = 20
c = a
x = b
if x < c:
	c = x
print(c)

#6. description: Factorial calculation
# code
n = 5
f = 1
x = n
while x > 0:
	f = f * x
	x = x - 1
print(f)

#7. description: Sum of first n natural numbers
# code
n = 10
s = 0
x = 1
while x <= n:
	s = s + x
	x = x + 1
print(s)

#8. description: Check if a number is prime
# code
n = 17
p = 1
x = 2
while x < n:
	if n / x == n // x:
		p = 0
	x = x + 1
print(p)

#9. description: Count digits in a number
# code
n = 12345
c = 0
x = n
while x > 0:
	c = c + 1
	x = x / 10
print(c)

#10. description: Square of a number
# code
a = 5
b = a * a
print(b)

#11. description: Cube of a number
# code
a = 3
b = a * a * a
print(b)

#12. description: Power of 2
# code
a = 4
b = 1
x = a
while x > 0:
	b = b * 2
	x = x - 1
print(b)

#13. description: Check if number is even
# code
a = 10
b = 0
if a / 2 == a // 2:
	b = 1
print(b)

#14. description: Check if number is odd
# code
a = 11
b = 0
if a / 2 != a // 2:
	b = 1
print(b)

#15. description: Sum of digits
# code
n = 123
s = 0
x = n
while x > 0:
	s = s + (x % 10)
	x = x // 10
print(s)

#16. description: Reverse a number
# code
n = 12345
r = 0
x = n
while x > 0:
	r = r * 10 + (x % 10)
	x = x // 10
print(r)

#17. description: Check palindrome number
# code
n = 12321
r = 0
x = n
y = n
while x > 0:
	r = r * 10 + (x % 10)
	x = x // 10
print(r == y)

#18. description: Find GCD (Euclidean algorithm)
# code
a = 48
b = 18
x = a
y = b
while y > 0:
	z = x % y
	x = y
	y = z
print(x)

#19. description: Armstrong number check
# code
n = 153
s = 0
x = n
y = n
while x > 0:
	d = x % 10
	s = s + (d * d * d)
	x = x // 10
print(s == y)

#20. description: Perimeter of a rectangle
# code
l = 12
w = 4
p = 2 * l
p = p + (2 * w)
print(p)


#21. description: Calculate simple interest
# code
p = 1000
r = 5
t = 2
s = p * r * t / 100
print(s)

#22. description: Calculate absolute value
# code
a = -10
b = a
if a < 0:
	b = a * -1
print(b)

#23. description: Check divisibility by 3 (through the sum of the digits)
# code
n = 27
s = 0
x = n
while x > 0:
	s = s + (x % 10)
	x = x // 10
b = 0
if s / 3 == s // 3:
	b = 1
print(b)

#24. description: Simple number pattern
# code
n = 5
x = 1
while x <= n:
	y = 1
	while y <= x:
		y = y + 1
	x = x + 1
print(x)

#25. description: Count positive numbers
# code
a = 5
b = -3
c = 0
x = a
if x > 0:
	c = c + 1
x = b
if x > 0:
	c = c + 1
print(c)

#26. description: Compute absolute difference
# code
a = 10
b = 7
c = a - b
x = b - a
if x > c:
	c = x
print(c)

#27. description: Check if square
# code
a = 16
b = 0
x = 1
while x * x <= a:
	if x * x == a:
		b = 1
	x = x + 1
print(b)

#28. description: Find the sum of all even numbers between 1 and n
# code
a = 2
b = 20
c = 0
while a <= b:
	c = c + a
	a = a + 2
print(c)

#29. description: Product of digits
# code
n = 123
a = 1
x = n
while x > 0:
	a = a * (x % 10)
	x = x // 10
print(a)

#30. description: Find the product of all odd numbers between 1 and n
# code
a = 1
b = 9  # Upper limit
c = 1
while a <= b:
	c = c * a
	a = a + 2
print(c)

#31. description: Simple temperature conversion from celsius to fahrenheit
# code
c = 100
f = c * 9 / 5 + 32
print(f)

#32. description: Decimal to binary conversion
# code
n = 10
a = 0
x = 1
while n > 0:
	a = a + (n % 2) * x
	n = n // 2
	x = x * 10
print(a)

#33. description: Count number of zeros in number
# code
n = 1030
a = 0
x = n
while x > 0:
	if x % 10 == 0:
		a = a + 1
	x = x // 10
print(a)

#34. description: Check ascending digits (from MSD to LSD)
# code
n = 1234
a = 1
x = n
y = x % 10
x = x // 10
while x > 0:
	z = x % 10
	if z > y:
		a = 0
	y = z
	x = x // 10
print(a)

#35. description: Least common multiple
# code
a = 12
b = 15
m = a
while m % b != 0:
	m = m + a
print(m)

#36. description: Count frequency of a digit
# code
n = 11211
d = 1
a = 0
x = n
while x > 0:
	if x % 10 == d:
		a = a + 1
	x = x // 10
print(a)

#37. description: sum of squares of first n natural numbers
# code
n = 5
s = 0
i = 1
while i <= n:
	s = s + i * i
	i = i + 1
print(s)

#38. description: Compute power without exponent
# code
a = 2
b = 3
c = 1
x = b
while x > 0:
	c = c * a
	x = x - 1
print(c)

#39. description: Find closest number
# code
a = 10
b = 15
c = 12
x = a - c
y = b - c
if x < 0:
	x = x * -1
if y < 0:
	y = y * -1
z = a
if y < x:
	z = b
print(z)

#40. description: Sum of first n odd numbers
# code
n = 5
a = 0
x = 1
while x <= n:
	a = a + (2 * x - 1)
	x = x + 1
print(a)

#41. description: Check if number ends with 5
# code
n = 125
a = 0
if n % 10 == 5:
	a = 1
print(a)

#42. description: Compute average of two numbers
# code
a = 10
b = 20
c = a + b
d = c / 2
print(d)

#43. description: Find maximum of three numbers
# code
a = 5
b = 15
c = 10
x = a
if b > x:
	x = b
if c > x:
	x = c
print(x)

#44. description: Generate multiplication table of a number
# code
n = 3
i = 1
while i <= 10:
	p = n * i
	print(p)
	i = i + 1

#45. description: Count numbers divisible by 2
# code
n = 10
a = 0
x = 1
while x <= n:
	if x / 2 == x // 2:
		a = a + 1
	x = x + 1
print(a)

#46. description: Check if a number is positive
# code
n = -10
a = 1
if n < 0:
	a = 0
print(a)

#47. description: Find the smallest divisor of a number greater than 1
# code
a = 49
b = 2
c = 0
while b <= a:
	if a % b == 0:
		if c == 0:
			c = b
	b = b + 1
print(c)

#48. description: Compute cube root (approximation)
# code
n = 27
a = 1
x = a * a * a
while x <= n:
	a = a + 1
	x = a * a * a
print(a - 1)

#49. description: Find the last non-zero digit (starting from the left i.e. first non-zero digit from the right)
# code
n = 120
a = 0
x = n
while x > 0:
	d = x % 10
	if d != 0:
		if a == 0:
			a = d
	x = x // 10
print(a)

#50. description: Decimal to octal conversion
# code
n = 16
a = 0
x = 1
while n > 0:
	d = n % 8
	a = a + d * x
	n = n // 8
	x = x * 10
print(a)

#51. description: Check if sum of digits is even
# code
n = 123
a = 0
s = 0
x = n
while x > 0:
	s = s + (x % 10)
	x = x // 10
if s / 2 == s // 2:
	a = 1
print(a)

#52. description: Compute factorial of sum
# code
a = 3
b = 4
c = a + b
f = 1
x = c
while x > 0:
	f = f * x
	x = x - 1
print(f)

#53. description: Find nearest perfect square
# code
n = 30
a = 1
while a * a < n:
	a = a + 1
b = a - 1
x = a * a
y = b * b
if n - y < x - n:
	x = y
print(x)

#54. description: Count numbers less than n
# code
n = 10
a = 0
x = 1
while x < n:
	a = a + 1
	x = x + 1
print(a)

#55. description: Check if number is a power of 2
# code
n = 16
a = 0
x = n
while x > 1:
	if x / 2 != x // 2:
		a = 1
	x = x / 2
print(a == 0)

#56. description: Compute greatest power of 2 less than n
# code
n = 50
a = 1
x = 2
while x <= n:
	a = x
	x = x * 2
print(a)

#57. description: Count occurrences of a digit in a number
# code
a = 1232
b = 2
c = 0
while a > 0:
    if a % 10 == b:
        c = c + 1
    a = a // 10
print(c)

#58. description: Count number of digits greater than 5
# code
n = 12657
a = 0
x = n
while x > 0:
	d = x % 10
	if d > 5:
		a = a + 1
	x = x // 10
print(a)

#59. description: Find second largest digit
# code
n = 1234
a = 0
b = 0
x = n
while x > 0:
	d = x % 10
	if d > a:
		a = d
	elif d > b:
		b = d
	x = x // 10
print(b)

#60. description: Check if number is magic number (sum of digits is 1)
# code
n = 10
a = 0
x = n
s = 0
while x > 0:
	s = s + (x % 10)
	x = x // 10
if s == 1:
	a = 1
print(a)

#61. description: Compute weighted sum of digits
# code
n = 123
a = 0
x = n
w = 1
while x > 0:
	d = x % 10
	a = a + d * w
	x = x // 10
	w = w + 1
print(a)

#62. description: Compute the sum of a number with its reversed
# code
a = 1234
s = a
b = 0
while a > 0:
    c = a % 10
    b = b * 10 + c
    a = a // 10
b = b + s
print(b)

#63. description: Check abundant number
# code
n = 12
a = 0
x = 1
s = 0
while x < n:
	if n / x == n // x:
		s = s + x
	x = x + 1
if s > n:
	a = 1
print(a)

#64. description: Compute digital root
# code
n = 456
a = n
while a > 9:
	x = a
	s = 0
	while x > 0:
		s = s + (x % 10)
		x = x // 10
	a = s
print(a)

#65. description: Find first digit
# code
n = 9345
a = n
while a > 9:
	a = a // 10
print(a)

#66. description: Count numbers less than digit sum
# code
n = 25
a = 0
x = 1
s = 0
y = n
while y > 0:
	s = s + (y % 10)
	y = y // 10
while x < s:
	a = a + 1
	x = x + 1
print(a)

#67. description: Check if number is cyclic (not quite ...)
# code
n = 142857
a = 0
x = n
y = n
r = 0
while x > 0:
	r = r * 10 + (x % 10)
	x = x // 10
z = y * 2
if r == z:
	a = 1
print(a)

#68. description: Compute product of non-zero digits
# code
n = 1023
a = 1
x = n
while x > 0:
	d = x % 10
	if d > 0:
		a = a * d
	x = x // 10
print(a)

#69. description: Find nearest number divisible by 5
# code
n = 22
a = n
b = n
while a / 5 != a // 5:
	a = a + 1
while b / 5 != b // 5:
	b = b - 1
x = a - n
y = n - b
z = a
if y < x:
	z = b
print(z)

#70. description: Check if number is a Harshad number
# code
n = 18
a = 0
x = n
s = 0
while x > 0:
	s = s + (x % 10)
	x = x // 10
if n / s == n // s:
	a = 1
print(a)

#71. description: Compute alternating sum of digits
# code
n = 1234
a = 0
x = n
s = 1
while x > 0:
	d = x % 10
	a = a + d * s
	x = x // 10
	s = s * -1
print(a)

#72. description: Find closest prime number
# code
n = 20
a = n
b = n
x = 2
while x < a:
	if a / x == a // x:
		a = a + 1
		x = 2
	else:
		x = x + 1
x = 2
while x < b:
	if b / x == b // x:
		b = b - 1
		x = 2
	else:
		x = x + 1
c = a - n
d = n - b
z = a
if d < c:
	z = b
print(z)

#73. description: Check if number is semiprime
# code
n = 15
a = 0
p = 0
x = 2
while x < n:
	if n / x == n // x:
		y = n / x
		p = p + 1
	x = x + 1
if p == 2:
	a = 1
print(a)

#74. description: Compute sum of reciprocals up to n
# code
n = 3
a = 0
x = 1
while x <= n:
	a = a + (1 / x)
	x = x + 1
print(a)

#75. description: Find first repeating digit
# code
n = 323174
a = 0
x = n
y = 0
while x > 0:
	d = x % 10
	z = y
	while z > 0:
		if d == z % 10:
			a = d
		z = z // 10
	y = y * 10 + d
	x = x // 10
print(a)

#76. description: Check if number is a perfect number
# code
n = 6
a = 0
s = 0
x = 1
while x < n:
	if n / x == n // x:
		s = s + x
	x = x + 1
if s == n:
	a = 1
print(a)

#77. description: Compute alternating factorial
# code
n = 4
a = 1
x = 1
s = 1
while x <= n:
	f = 1
	y = x
	while y > 0:
		f = f * y
		y = y - 1
	a = a + f * s
	x = x + 1
	s = s * -1
print(a)

#78. description: Find number of zeros at end of factorial
# code
n = 5
a = 0
x = 5
while x <= n:
	y = x
	while y / 5 == y // 5:
		a = a + 1
		y = y / 5
	x = x + 1
print(a)

#79. description: Check if number is a triangular number
# code
n = 10
a = 0
x = 1
s = 0
while s < n:
	s = s + x
	x = x + 1
if s == n:
	a = 1
print(a)

#80. description: Compute sum of reciprocal factorial
# code
n = 5
a = 0
x = 0
while x <= n:
	f = 1
	y = x
	while y > 0:
		f = f * y
		y = y - 1
	a = a + (1 / f)
	x = x + 1
print(a)

#81. description: Compute alternating sum
# code
n = 5
a = 0
x = 1
s = 1
while x <= n:
	a = a + x * s
	x = x + 1
	s = s * -1
print(a)

#82. description: Check if number is a Kaprekar number
# code
n = 45
a = 0
x = n * n
y = n
t = 0
while y > 0:
	t = t + 1
	y = y // 10
s = 0
b = 1
while t > 0:
	s = s + (x % 10) * b
	x = x // 10
	t = t - 1
	b = b * 10
r = s + x
if r == n:
	a = 1
print(a)

#83. description: Compute geometric progression sum
# code
a = 2
r = 3
n = 4
s = 0
x = 0
while x < n:
	p = a
	y = x
	while y > 0:
		p = p * r
		y = y - 1
	s = s + p
	x = x + 1
print(s)

#84. description: Find first digit starting from the left, that is different than the LSD
# code
n = 32213
a = 0
x = n
y = x % 10
x = x // 10
while x > 0:
	z = x % 10
	if z != y:
		a = z
	x = x // 10
print(a)

#85. description: Sieve of Eratosthenes to find all prime numbers up to n
# code
n = 15
p = 2
while p <= n:
	a = 1
	i = 2
	while i < p:
		if p % i == 0:
			a = 0
		i = i + 1
	if a == 1:
		print(p)
	p = p + 1

#86. description: Compute Euler's totient function
# code
n = 9
a = n
x = 2
while x * x <= n:
    if n % x == 0: 
        while n % x == 0: 
            n = n // x
        a = a - a // x
    x = x + 1
if n > 1:
    a = a - a // n
print(a)

#87. description: Find number of trailing zeros in factorial
# code
n = 10
a = 0
b = 1
if n == 0:
	print(a)
elif n == 1:
	print(b)
else:
	x = 2
	while x <= n:
		c = a + b
		a = b
		b = c
		x = x + 1
	print(b)

#88. description: Check if number = sum of its digits elevated to d where d == number of digits in the number"
# code
n = 153
a = 0
x = n
s = 0
d = 0
while x > 0:
	d = d + 1
	x = x // 10
x = n
while x > 0:
	y = x % 10
	z = 1
	w = d
	while w > 0:
		z = z * y
		w = w - 1
	s = s + z
	x = x // 10
if s == n:
	a = 1
print(a)

#89. description: Compute sum of prime factors
# code
n = 24
a = 0
x = 2
while x <= n:
	if n % x == 0:
		y = x
		p = 1
		z = 2
		while z < y:
			if y % z == 0:
				p = 0
			z = z + 1
		if p == 1:
			a = a + x
	x = x + 1
print(a)

#90. description: Find largest digit less than given digit
# code
n = 5274
d = 9
a = 0
x = n
while x > 0:
	y = x % 10
	if y < d:
		if y > a:
			a = y
	x = x // 10
print(a)

#91. description: Check if number is a smooth number
# code
n = 24
k = 5
a = 0
x = 2
m = 0
while x <= n:
	if n % x == 0:
		if x > k:
			m = 1
		n = n // x
	else:
		x = x + 1
if m == 0:
	a = 1
print(a)

#92. description: Compute sum of digital roots
# code
n = 123
a = 0
x = 1
while x <= n:
	y = x
	s = 0
	while y > 0:
		s = s + (y % 10)
		y = y // 10
	while s > 9:
		y = s
		s = 0
		while y > 0:
			s = s + (y % 10)
			y = y // 10
	a = a + s
	x = x + 1
print(a)

#93. description: Find smallest number with same digit sum
# code
n = 123
a = n
s = 0
x = n
while x > 0:
	s = s + (x % 10)
	x = x / 10
x = n - 1
while x > 0:
	y = x
	t = 0
	while y > 0:
		t = t + (y % 10)
		y = y / 10
	if t == s:
		a = x
	x = x - 1
print(a)

#94. description: Check if number is a palindromic prime
# code
n = 131
a = 0
p = 1
x = 2
while x < n:
	if n % x == 0:
		p = 0
	x = x + 1
if p == 1:
	x = n
	r = 0
	while x > 0:
		r = r * 10 + (x % 10)
		x = x // 10
	if r == n:
		a = 1
print(a)

#95. description: Compute sum of aliquot divisors
# code
n = 12
a = 0
x = 1
while x < n:
	if n % x == 0:
		a = a + x
	x = x + 1
print(a)

#96. description: Checking if two numbers are rep-digits
# code
n = 222
b = n % 10
a = 1
x = n
while x > 0:
	d = x % 10
	if d != b:
		a = 0
	x = x // 10
print(a)
n = 112
b = n % 10
a = 1
x = n
while x > 0:
	d = x % 10
	if d != b:
		a = 0
	x = x // 10
print(a)

#97. description: Concatenate two numbers n|m
# code
n = 91
m = 123
b = 10
x = m // 10
while x > 0:
	b = b * 10
	x = x // 10
n = n * b
a = n + m
print(a)

#98. description: Compute sum of nth power of digits
# code
n = 89
a = 3
x = n
s = 0
while x > 0:
	d = x % 10
	y = 1
	z = a
	while z > 0:
		y = y * d
		z = z - 1
	s = s + y
	x = x // 10
print(s)

#99. description: Find largest cyclic number less than m
# code
m = 100
y = m
r = 0
while y > 0:
	n = y
	a = n
	x = 2
	while x <= n:
		if n % x == 0:
			while n % x == 0:
				n = n // x
			a = a - a // x
		x = x + 1
	if n > 1:
		a = a - a // n
	if a > y:
		i = a
		j = y
	else:
		i = y
		j = a
	while j > 0:
		z = i % j
		i = j
		j = z
	if i == 1:
		if r == 0:
			r = y
	y = y - 1
print(r)