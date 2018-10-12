#!/usr/bin/bash/env bash
isNew = "N"

echo 본 파일은 두번 설치하면 안됩니다. 확실히 Databuzz tech blog를 처음 설치하시는거 맞나요[Y/N]?
read isNew

if [ ${isNew} = "Y" ] || [ ${isNew} = "y" ]; then
  mkdir scaffolds
  cp scaffolds_/draft.md scaffolds
  cp scaffolds_/page.md scaffolds
  cp scaffolds_/post.md scaffolds
  npm install
  cd themes
  rm -rf hexo-theme-repo
  git clone https://github.com/Databuzz-team/hexo-theme-repo.git
  cd ..
else
  echo "Okay bye."
fi
