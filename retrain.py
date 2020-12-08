import subprocess

c=0
for i in range(1):
    subprocess.call(['python3', 'train.py'])
    c+=4
    print(f'---------------------------------------{c} episodes over---------------------------------------')