#!/usr/bin/bash/env bash
cd themes/hexo-theme-repo/
git pull origin master
cd ..
cd ..
git pull origin master
npm set audit false
npm install
hexo clean
hexo generate -d
git add .
git commit -m "Added new posts"
git push origin master
