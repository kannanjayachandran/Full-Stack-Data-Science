# Version Control

Git is the version control I use. It is a distributed version control system, which means that each user has a complete copy of the repository. This is in contrast to a centralized version control system, where there is a single repository that everyone works on. It is currently the most popular version control system and is used by most open source projects. 

## Setting up Git

Install git for your OS and get an account on GitHub. Then in your machine open the `.gitconfig` file in your home directory and add the following lines:

```bash
[user]
    name = {Your Name}
    email = {Your Email}

[credential]
    helper = cache --timeout=3600

[alias]
    co = checkout
    ci = commit
    st = status
    br = branch
    hist = log --pretty=format:\"%h %ad | %s%d [%an]\" --graph --date=short

[color]
    branch = auto
    diff = auto
    interactive = auto
    status = auto
```
> These are my configs, you can change them as you like.

## [GIT Online Documentation](https://git-scm.com/book/en/v2)