

def f1_score_calculation(recall, precision):
	result = recall * precision * 2 / (precision + recall)
	return result

def test():
	return
	
def test_var_args (f_arg, argv):
	print("first normal arg: ", f_arg)
	for arg in argv:
		print("another arg through *argv: ", arg)
		
test_var_args('python', 'eggs', 'ready')




print(f1_score_calculation(95.55, 95.57))
print(f1_score_calculation(92.75, 92.77))
print(f1_score_calculation(94.21, 94.23))
print(f1_score_calculation(95.63, 95.64))
print(f1_score_calculation(65, 35))
print(f1_score_calculation(0.94351042367182247, 0.94323655353861802))

