---
title: <Hexo> 블로그에 수식 사용하기 - mathjax 설정
categories:
  - HyeShin Oh
tags:
  - Hexo
  - Mathjax
date: 2018-10-13 14:30:55
thumbnail:
---

필자의 개인 블로그에 포스팅한 내용을 가져와 소개합니다. ([원문 블로그](https://hyeshinoh.github.io/2018/10/24/hexo_mathjax_00/))


Hexo 블로그에서 LaTex로 수식을 작성할 수 있도록 mathjax를 설정하는 방법을 정리해보겠습니다.


## 1. 설치

### 1) renderer 설치 및 세팅

Hexo의 기본 renderer인 hexo-renderer-marked는 mathjax 문법을 지원하지 않는다고 합니다. 따라서 다음과 같이 새로운 rendering engine으로 교체해줍니다.

`$ npm uninstall hexo-renderer-marked --save`
`$ npm install hexo-renderer-kramed --save`


그리고 `<your-project-dir>/node_modules/hexo-reneder-kramed/lib/renderer.js`를 열어 다음과 같이 return 값을 text로 수정합니다.

```javascript
// Change inline math rule
function formatText(text) {
  // Fit kramed's rule: $$ + \1 + $$
  // return text.replace(/`\$(.*?)\$`/g, '$$$$$1$$$$');
  return text;
}
```
### 2) mathjax 설치

다음으로는 mathjax plugin을 설치합니다.
`npm install hexo-renderer-mathjax --save`

그리고 `<your-project-dir>/node_modules/hexo-reneder-mathjax/mathjax.html`을 열고 CDN URL을 아래와 같이 수정합니다.

```html
<!-- <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script> -->
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'></script>
```


## 2. LaTex와 markdown의 문법 충돌 fix하기
LaTex와 markdown에는 다음과 같이 문법이 충돌하는 부분이 있습니다. 
- markdown: `*`과 `_`는 bold와 italic
- LaTex: `_`는 subscript

따라서 `_`는 LaTex의 문법만을 따라서 아랫첨자를 나타내도록 하기위해`node_modules\kramed\lib\rules\inline.js`를 열고 다음과 같이 수정합니다.

```
escape: /^\\([`*\[\]()#$+\-.!_>])/,
em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```

## 3. Mathjax 사용하기

사용하고 있는 theme의 `_config.yml` 파일을 열고 다음과 같이 mathjax를 enabling 해줍니다.

```
mathjax:
  enable: true
```

## 4. markdown post 작성하기

이제 hexo 블로그에 수식을 사용하기 위한 설정은 모두 마쳤습니다.
마지막으로 post 작성시 header 부분에 `mathjax: true`를 넣어주면 블로그에 수식이 잘 표현되게 됩니다.



#### 참고 자료
- 블로그 [Make Hexo Support Latex](https://www.infiniteft.xyz/2018/03/21/make-hexo-support-latex/)
- 블로그 [hexo-inline-math](https://irongaea.github.io/2018/08/21/hexo-inline-math/)
- 블로그 [MathJax로 LaTeX 사용하기](https://johngrib.github.io/wiki/mathjax-latex/#3-%EB%8F%84%EA%B5%AC)
- [www.mathjax.org](https://www.mathjax.org/#gettingstarted)

