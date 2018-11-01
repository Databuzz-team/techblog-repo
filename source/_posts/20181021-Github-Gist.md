---
title: <Github> Gist를 사용하여 Jupyter Notebook 포스팅하기
categories:
  - Danial Nam
tags:
  - Git
date: 2018-10-21 22:01:32
thumbnail:
---
##### [Gist](https://gist.github.com/)를 사용하면 아래처럼 소스코드를 임베딩 할 수 있다.

아래의 예는 **Jupyter Notebook** 을 임베딩한 것이지만, 이 외에도 .py, .md, .html 등 소스 코드는 다 할 수 있다.

<script src="https://gist.github.com/DanialDaeHyunNam/afc48b2814cd7798ee7dbaa00e321468.js"></script>

<h5 style='color: red;'>
  만약 위의 박스 안에 Jupyter Notebook이 보이지 않는다면.. Gist 사이트에 올린 .ipynb 파일이 Rendering 실패하는 것이므로 <a href='#new-method'>여기</a>를 클릭해서 불필요한 내용은 Skip 하시길..
</h5>

사용하는 방법은 아래와 같다.

---

### 1. 먼저 [Gist](https://gist.github.com/) 사이트로 이동
<img src="/images/danial/gist_1.png">
위의 사진처럼 코드를 적는 창이 있다. Jupyter Notebook의 경우에는 해당 파일을 **Drag & Drop** 하면 된다.

### 2. Indent Mode를 Spaces에서 Tabs로 변경
<img src="/images/danial/gist_2.png">
이 경우는 **Jupyter Notebook** 을 임베딩하는 경우에만 적용될 수도 있는 사항이니 조심하자.

필자는 Jupyter Notebook을 처음부터 도전하였는데, Spaces의 경우에는 계속 Render를 실패하기에 이유를 찾아보다가 발견한 방법이다.

### 3. Script Copy
<img src="/images/danial/gist_3.png">
이미지처럼 Embed 옆에 아이콘을 클릭하면 아래처럼 **script** 태그가 복사된다. Markdown 파일이나 HTML에 붙여넣기하면 된다.
```html
<script src="https://gist.github.com/{UserId}/{script}.js"></script>
```

### Tip 1) 이후에 다른 소스코드를 포스팅하고자 하면 Add file이 아닌 New gist로 새로운 파일들을 추가하면 된다.
<img src="/images/danial/gist_helper.png">

### Tip 2) 이미 Gist에서 정해준 iframe의 크기를 조절하고 싶은 경우에는 아래의 css를 추가해주면 된다.
```css
.gist{
  max-width: 80%;
  margin-top: 10px;
}

.gist-data{
  max-height: 300px;
}

.gist iframe.render-viewer{
  max-height: 260px;
}
```

<h5 style='color: blue;'>
  Gist가 Rendering을 문제없이 해냈다면 여기서부터 아래 내용은 볼 필요가 없다.
</h5>

---

<h3 id='new-method' href="#new-method">
  New) Gist에서 제공하는 방법보다는 불편하고 깔끔하지는 않지만, Jupyter Notebook을 Embed하는 다른 방법이 있어 소개하고자 한다.
</h3>

> 이 방법의 경우에는 사실 github repo에 올린 jupyter notebook에도 적용되는 방법이므로, 꼭 Gist를 사용할 필요는 없지만 이 포스트에서는 Gist에 올린 경우에 대해서만 설명하겠다.

### 1. Gist 사이트에 업로드
일단 위의 순서 중에 <a id="1-먼저-Gist-사이트로-이동" href='#1-먼저-Gist-사이트로-이동'>1.Gist 사이트로 이동</a>는 필요하므로 Drag&Drop 하는 부분까지는 진행을 하자.

### 2. 업로드한 Gist의 주소를 복사
1번에서 업로드한 Gist의 URL을 복사한다.
```html
https://gist.github.com/{id}/{key-value}
```

### 3. https://nbviewer.jupyter.org/ 로 이동
[nbviewer 사이트](https://nbviewer.jupyter.org/)로 이동을 하면,
**URL | GitHub username | GitHub username/repo | Gist ID** 를 입력하는 Input 창이 있다. 거기에 2번에서 복사한 주소를 넣으면 **해당 Notebook이 Rendering** 된 화면으로 넘어간다.

### 4. Hexo Tag Plugins를 이용해서 Embedding하면 끝!
필자는 아래의 Markdown에 html을 같이 사용했다.
```Markdown
<div class='notebook-embedded'>
{% iframe https://nbviewer.jupyter.org/gist/{id}/{unknown-value} 100% 100% %}
</div>
```
여기서 굳이 [Hexo Tag Plugins](https://hexo.io/docs/tag-plugins.html)에서 제공한
```
{% iframe [width] [height] %}
```
방식의 표현식을 사용한 것은 Markdown preview에서 iframe이 계속해서 reload되는 것이 싫어서 그랬을 뿐 html 태그를 사용해도 무방하다.

div에 class를 준 것은 원하는 디자인 틀을 만들기 위해서였고 css는 아래와 같다.
```css
.notebook-embedded{
  width: 100%;
  border: 1px solid #eee;
  border-bottom: 40px solid #eee;
  border-radius: 4px;
  height: 400px;
}
```

### 5. 적용한 결과는 아래와 같다
<div class='notebook-embedded'>
{% iframe https://nbviewer.jupyter.org/gist/DanialDaeHyunNam/afc48b2814cd7798ee7dbaa00e321468 100% 100% %}
</div>
