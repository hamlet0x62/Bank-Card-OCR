(function(e){function t(t){for(var r,a,s=t[0],u=t[1],c=t[2],i=0,p=[];i<s.length;i++)a=s[i],o[a]&&p.push(o[a][0]),o[a]=0;for(r in u)Object.prototype.hasOwnProperty.call(u,r)&&(e[r]=u[r]);f&&f(t);while(p.length)p.shift()();return l.push.apply(l,c||[]),n()}function n(){for(var e,t=0;t<l.length;t++){for(var n=l[t],r=!0,a=1;a<n.length;a++){var s=n[a];0!==o[s]&&(r=!1)}r&&(l.splice(t--,1),e=u(u.s=n[0]))}return e}var r={},a={app:0},o={app:0},l=[];function s(e){return u.p+"static/js/"+({about:"about"}[e]||e)+"."+{about:"05f6d4e3"}[e]+".js"}function u(t){if(r[t])return r[t].exports;var n=r[t]={i:t,l:!1,exports:{}};return e[t].call(n.exports,n,n.exports,u),n.l=!0,n.exports}u.e=function(e){var t=[],n={about:1};a[e]?t.push(a[e]):0!==a[e]&&n[e]&&t.push(a[e]=new Promise(function(t,n){for(var r="static/css/"+({about:"about"}[e]||e)+"."+{about:"5799640d"}[e]+".css",o=u.p+r,l=document.getElementsByTagName("link"),s=0;s<l.length;s++){var c=l[s],i=c.getAttribute("data-href")||c.getAttribute("href");if("stylesheet"===c.rel&&(i===r||i===o))return t()}var p=document.getElementsByTagName("style");for(s=0;s<p.length;s++){c=p[s],i=c.getAttribute("data-href");if(i===r||i===o)return t()}var f=document.createElement("link");f.rel="stylesheet",f.type="text/css",f.onload=t,f.onerror=function(t){var r=t&&t.target&&t.target.src||o,l=new Error("Loading CSS chunk "+e+" failed.\n("+r+")");l.code="CSS_CHUNK_LOAD_FAILED",l.request=r,delete a[e],f.parentNode.removeChild(f),n(l)},f.href=o;var d=document.getElementsByTagName("head")[0];d.appendChild(f)}).then(function(){a[e]=0}));var r=o[e];if(0!==r)if(r)t.push(r[2]);else{var l=new Promise(function(t,n){r=o[e]=[t,n]});t.push(r[2]=l);var c,i=document.createElement("script");i.charset="utf-8",i.timeout=120,u.nc&&i.setAttribute("nonce",u.nc),i.src=s(e),c=function(t){i.onerror=i.onload=null,clearTimeout(p);var n=o[e];if(0!==n){if(n){var r=t&&("load"===t.type?"missing":t.type),a=t&&t.target&&t.target.src,l=new Error("Loading chunk "+e+" failed.\n("+r+": "+a+")");l.type=r,l.request=a,n[1](l)}o[e]=void 0}};var p=setTimeout(function(){c({type:"timeout",target:i})},12e4);i.onerror=i.onload=c,document.head.appendChild(i)}return Promise.all(t)},u.m=e,u.c=r,u.d=function(e,t,n){u.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:n})},u.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},u.t=function(e,t){if(1&t&&(e=u(e)),8&t)return e;if(4&t&&"object"===typeof e&&e&&e.__esModule)return e;var n=Object.create(null);if(u.r(n),Object.defineProperty(n,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var r in e)u.d(n,r,function(t){return e[t]}.bind(null,r));return n},u.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return u.d(t,"a",t),t},u.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},u.p="/",u.oe=function(e){throw console.error(e),e};var c=window["webpackJsonp"]=window["webpackJsonp"]||[],i=c.push.bind(c);c.push=t,c=c.slice();for(var p=0;p<c.length;p++)t(c[p]);var f=i;l.push([0,"chunk-vendors"]),n()})({0:function(e,t,n){e.exports=n("56d7")},"034f":function(e,t,n){"use strict";var r=n("64a9"),a=n.n(r);a.a},"2eb2":function(e,t,n){e.exports=n.p+"static/img/7.3a8f7336.jpeg"},"56d7":function(e,t,n){"use strict";n.r(t);n("cadf"),n("551c"),n("f751"),n("097d");var r=n("2b0e"),a=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",{attrs:{id:"app"}},[n("el-header",{staticClass:"nav"},[n("Nav")],1),n("router-view")],1)},o=[],l=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("el-menu",{attrs:{"default-active":"/",mode:"horizontal"},on:{select:e.handleSelect}},[n("el-menu-item",{attrs:{index:"/"}},[e._v("\n        主页\n    ")]),n("el-menu-item",{attrs:{index:"/about"}},[e._v("\n        帮助\n    ")])],1)},s=[],u={methods:{handleSelect:function(e){this.$router.push(e)}}},c=u,i=n("2877"),p=Object(i["a"])(c,l,s,!1,null,null,null),f=p.exports,d={name:"app",components:{Nav:f}},h=d,g=(n("034f"),Object(i["a"])(h,a,o,!1,null,null,null)),m=g.exports,v=n("8c4f"),b=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("el-row",{attrs:{type:"flex",justify:"center",width:"100%"}},[n("el-col",{attrs:{span:24}},[n("ResultView")],1)],1)},y=[],w=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("el-menu",{attrs:{mode:"vertical"}},[n("el-menu-item",[e._v("\n        Detection\n    ")]),n("el-menu-item",[e._v("\n        Recognize\n    ")])],1)},x=[],_={},C=_,j=Object(i["a"])(C,w,x,!1,null,null,null),$=j.exports,S=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("el-row",{attrs:{type:"flex",justify:"center"}},[n("el-col",{directives:[{name:"loading",rawName:"v-loading",value:e.loading,expression:"loading"}],staticClass:"resultContainer",attrs:{span:18}},[e.hasFinalResult?n("div",[n("el-tooltip",{attrs:{placement:"bottom"}},[n("span",{attrs:{slot:"content"},slot:"content"},[e._v("复制"+e._s(this.copied?"成功":""))]),n("h3",{on:{click:e.copyRst}},[e._v("识别结果: "+e._s(this.recognizeResult.join(" ")))])])],1):n("h3",[e._v("请点击下方的操作按钮")]),n("el-row",{attrs:{type:"flex",justify:"center"}},[n("el-col",{attrs:{span:3}},[n("el-upload",{attrs:{action:"","show-file-list":!1,"auto-upload":!1},nativeOn:{change:function(t){return e.uploadImg(t)}}},[n("el-button",{attrs:{slot:"trigger",type:"info"},slot:"trigger"},[e._v("选择图片")])],1)],1),n("el-col",{attrs:{span:3}},[n("el-button",{attrs:{type:"primary",disabled:!e.hasPicture},on:{click:e.detect}},[e._v("检测卡号区域")])],1),n("el-col",{attrs:{span:3}},[n("el-button",{attrs:{type:"success",disabled:!e.hasCropped},on:{click:e.uploadCropImg}},[e._v("识别卡号")])],1)],1),n("el-row",{directives:[{name:"show",rawName:"v-show",value:e.hasPicture,expression:"hasPicture"}],attrs:{type:"flex",justify:"center"}},[n("el-col",{attrs:{span:18}},[n("div",{attrs:{id:"cropBox"}},[n("vue-cropper",{ref:"cropper",attrs:{guides:!0,"auto-crop":!0,"auto-crop-area":.95,rotatable:!0,background:!0,"img-style":{width:"640px",height:"480px"},src:this.curImgSrc}})],1)]),n("el-col",{attrs:{span:3}},[n("el-row",{staticStyle:{"margin-top":"250px"}},[n("el-col",{attrs:{span:18}},[n("el-input",{attrs:{placeholder:"旋转角度",inline:""},model:{value:e.rotateDegree,callback:function(t){e.rotateDegree=t},expression:"rotateDegree"}})],1),n("el-col",{attrs:{span:3}},[n("el-button",{attrs:{type:"warning",icon:"el-icon-refresh-right",circle:""},on:{click:e.rotate}})],1)],1)],1)],1)],1)],1)},D=[],R=(n("a481"),n("6762"),n("2fdb"),"http://127.0.0.1:5000"),O=("".concat(R,"/upload"),"".concat(R,"/recognize")),k="".concat(R,"/detect"),E=n("95c3"),P=n.n(E),N=n("bc3a"),I=n.n(N),A=/^.*base64,?(.*)$/,T={props:{imgSrc:{default:n("2eb2")},uploadActionUrl:{default:""},subject:{default:""}},data:function(){return{imgStyle:{height:"300px"},loading:!1,curImgSrc:null,imageHeight:null,imageWidth:null,recognizeResult:[],clipboard:null,copied:!1,rotateDegree:90,hasCropped:!1}},computed:{hasPicture:function(){return null!==this.curImgSrc},hasFinalResult:function(){return this.recognizeResult.length>0}},methods:{detect:function(){var e=this,t=this.$refs["cropper"].getCroppedCanvas().toDataURL(),n=A.exec(t);if(null!==n){var r=this.$refs["cropper"].getCropBoxData(),a=r["width"],o=r["height"],l=r["left"],s=r["top"];console.log(a,o),this.loading=!0,I.a.post(k,{b64:n[1]}).then(function(t){var n=t.data,r=n.xmin,u=n.ymin,c=n.xmax,i=n.ymax;n.score;console.log(t.data);var p=(i-u)*o,f=(c-r)*a;r=~~(r*a),u=~~(u*o),e.$refs["cropper"].setCropBoxData({left:r+l,top:u+s,height:p,width:f}),e.loading=!1,e.copied=!1,e.hasCropped=!0}).catch(function(t){console.log(t),e.$message.error("识别失败"),e.loading=!1})}},uploadImg:function(e){var t=this,n=e.target.files[0];if(console.log(n),n.type.includes("image"))if("function"===typeof FileReader){var r=new FileReader;r.onload=function(e){var n=e.target.result;t.curImgSrc=n,t.$refs["cropper"].replace(n),t.hasCropped=!1},r.readAsDataURL(n)}else this.$message.error("暂不支持的浏览器");else this.$message.warning("请选择图片文件后再进行上传")},rotate:function(){this.$refs["cropper"].rotate(this.rotateDegree);var e=this.$refs["cropper"].getCanvasData();console.log(this.$refs["cropper"].getCanvasData()),this.$refs["cropper"].setCropBoxData({left:e["left"],top:e["top"],width:e["width"],height:e["height"]})},onCropMove:function(){},uploadCropImg:function(){var e=this,t=this.$refs["cropper"].getCroppedCanvas().toDataURL(),n=A.exec(t);n.length>0&&(console.log(n[0]),I()({method:"post",url:O,data:{b64:n[1]}}).then(function(t){var n=t.data;e.recognizeResult=n["result"]}).catch(function(e){console.log(e)}))},copyRst:function(){var e=this,t=this.recognizeResult.join("");this.$copyText(t).then(function(){return e.copied=!0}).catch(function(){e.$message.error("复制失败")})}},components:{"vue-cropper":P.a},created:function(){}},z=T,B=(n("806d"),Object(i["a"])(z,S,D,!1,null,null,null)),L=B.exports,F={name:"home",components:{Nav:f,AsideNav:$,ResultView:L},methods:{},data:function(){return{activeIndex:"Detect"}}},M=F,U=Object(i["a"])(M,b,y,!1,null,null,null),q=U.exports;r["default"].use(v["a"]);var H=new v["a"]({routes:[{path:"/",name:"home",component:q},{path:"/about",name:"about",component:function(){return n.e("about").then(n.bind(null,"f820"))}}]}),J=n("2f62");r["default"].use(J["a"]);var V=new J["a"].Store({state:{},mutations:{},actions:{}}),K=n("4eb5"),W=n.n(K),G=(n("10cb"),n("450d"),n("f3ad")),Q=n.n(G),X=(n("0c67"),n("299c")),Y=n.n(X),Z=(n("be4f"),n("896a")),ee=n.n(Z),te=(n("f225"),n("89a9")),ne=n.n(te),re=(n("a673"),n("7b31")),ae=n.n(re),oe=(n("a769"),n("5cc3")),le=n.n(oe),se=(n("adec"),n("3d2d")),ue=n.n(se),ce=(n("de31"),n("c69e")),ie=n.n(ce),pe=(n("acb6"),n("c673")),fe=n.n(pe),de=(n("f4f9"),n("c2cc")),he=n.n(de),ge=(n("7a0f"),n("0f6c")),me=n.n(ge),ve=(n("8bd8"),n("4cb2")),be=n.n(ve),ye=(n("4ca3"),n("443e")),we=n.n(ye),xe=(n("1951"),n("eedf")),_e=n.n(xe),Ce=(n("7f7f"),n("0fb7"),n("f529")),je=n.n(Ce);r["default"].use(je.a.name,je.a),r["default"].use(_e.a),r["default"].use(we.a),r["default"].use(be.a),r["default"].use(me.a),r["default"].use(he.a),r["default"].use(fe.a),r["default"].use(ie.a),r["default"].use(ue.a),r["default"].use(le.a),r["default"].use(ae.a),r["default"].use(ne.a),r["default"].use(ee.a),r["default"].use(Y.a),r["default"].use(Q.a),r["default"].prototype.$message=je.a,r["default"].config.productionTip=!1,r["default"].use(W.a),new r["default"]({router:H,store:V,render:function(e){return e(m)}}).$mount("#app")},"64a9":function(e,t,n){},"806d":function(e,t,n){"use strict";var r=n("842c"),a=n.n(r);a.a},"842c":function(e,t,n){}});
//# sourceMappingURL=app.a3c821c7.js.map