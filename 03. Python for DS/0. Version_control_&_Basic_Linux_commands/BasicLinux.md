# Linux Basics

## Basic Linux Commands

1. **pwd** : Print Working Directory
2. **ls** : List directory contents
3. **cd** : Change directory
4. **mkdir** : Make directory
5. **touch** : Create file
```bash
touch file1.txt file2.txt file3.txt
```
6. **cp** : Copy file or directory 
```bash
cp from.txt folder
```
7. **mv** : Move file or directory
```bash
mv file1.txt folder1/
```
8. **rm** : Remove file or directory
```bash
rm file1.txt
```
9. **cat** : Concatenate files and print on the standard output
```bash
cat file1.txt
```
10. **head** : Output the first part of files
```bash
head file1.txt
```
11. **tail** : Output the last part of files
```bash
tail file1.txt
```
12. **grep** : Print lines matching a pattern
```bash
grep "pattern" file1.txt
```
13. **wc** : Print newline, word, and byte counts for each file
```bash
wc file1.txt
```
14. **vim** : Text Editor
```bash
vim file1.txt
```
15. **chmod** : Change the permission of a file or directory
```bash
# 777 means read, write, and execute permissions for the owner, group, and others.
# 666 means read and write permissions for the owner, group, and others.
# 444 means read-only permissions for the owner, group, and others.
# 00 means no permission for the owner, group, and others.

chmod 777 file1.txt
```

16. **chown** : Change the owner of a file or directory
```bash
chown user1 file1.txt
```
17. **chgrp** : Change the group of a file or directory
```bash
chgrp group1 file1.txt
```
18. **ls -l** : To show the permissions of a file or directory
```bash 
ls -l
```
19. **sudo** : Run command with the security privileges of the superuser
```bash
sudo apt-get install python3
```
20. **apt-get** : Command-line tool for handling packages
```bash
apt-get install python3
```