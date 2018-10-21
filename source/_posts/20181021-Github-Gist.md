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

사용하는 방법은 아래와 같다.
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


### Related Posts
