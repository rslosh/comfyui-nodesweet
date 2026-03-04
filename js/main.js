(function(){"use strict";try{if(typeof document<"u"){var a=document.createElement("style");a.appendChild(document.createTextNode(".ease-curve-root[data-v-a2535006]{display:flex;flex-direction:column;align-items:center;gap:4px;padding:4px;font-family:sans-serif}.curve-svg[data-v-a2535006]{width:100%;max-width:260px;border-radius:6px;-webkit-user-select:none;user-select:none;touch-action:none}.handle[data-v-a2535006]{cursor:grab}.handle[data-v-a2535006]:active{cursor:grabbing}.grid-toggle[data-v-a2535006]{background:#2a2a4a;color:#aab;border:1px solid #3a3a5a;border-radius:4px;padding:3px 10px;font-size:10px;cursor:pointer;width:100%;max-width:260px}.grid-toggle[data-v-a2535006]:hover{background:#3a3a5a;color:#dde}.preset-grid[data-v-a2535006]{display:grid;grid-template-columns:repeat(5,1fr);gap:3px;width:100%;max-width:260px}.preset-thumb[data-v-a2535006]{display:flex;flex-direction:column;align-items:center;cursor:pointer;border:1px solid #2a2a4a;border-radius:4px;padding:2px;background:#1a1a2e;transition:border-color .15s}.preset-thumb[data-v-a2535006]:hover{border-color:#6cf}.preset-thumb.active[data-v-a2535006]{border-color:#6cf;background:#224}.thumb-svg[data-v-a2535006]{width:100%;aspect-ratio:1}.thumb-label[data-v-a2535006]{font-size:6px;color:#778;text-align:center;line-height:1.1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;width:100%}")),document.head.appendChild(a)}}catch(e){console.error("vite-plugin-css-injected-by-js",e)}})();
import { app as Qr } from "../../scripts/app.js";
/**
* @vue/shared v3.5.29
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/
// @__NO_SIDE_EFFECTS__
function tt(e) {
  const t = /* @__PURE__ */ Object.create(null);
  for (const n of e.split(",")) t[n] = 1;
  return (n) => n in t;
}
const Y = process.env.NODE_ENV !== "production" ? Object.freeze({}) : {}, wt = process.env.NODE_ENV !== "production" ? Object.freeze([]) : [], ie = () => {
}, As = () => !1, en = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && // uppercase letter
(e.charCodeAt(2) > 122 || e.charCodeAt(2) < 97), bn = (e) => e.startsWith("onUpdate:"), Z = Object.assign, No = (e, t) => {
  const n = e.indexOf(t);
  n > -1 && e.splice(n, 1);
}, Xr = Object.prototype.hasOwnProperty, B = (e, t) => Xr.call(e, t), T = Array.isArray, gt = (e) => tn(e) === "[object Map]", Ms = (e) => tn(e) === "[object Set]", Jo = (e) => tn(e) === "[object Date]", M = (e) => typeof e == "function", X = (e) => typeof e == "string", qe = (e) => typeof e == "symbol", U = (e) => e !== null && typeof e == "object", bo = (e) => (U(e) || M(e)) && M(e.then) && M(e.catch), Rs = Object.prototype.toString, tn = (e) => Rs.call(e), Oo = (e) => tn(e).slice(8, -1), Fs = (e) => tn(e) === "[object Object]", xo = (e) => X(e) && e !== "NaN" && e[0] !== "-" && "" + parseInt(e, 10) === e, Bt = /* @__PURE__ */ tt(
  // the leading comma is intentional so empty string "" is also included
  ",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"
), Zr = /* @__PURE__ */ tt(
  "bind,cloak,else-if,else,for,html,if,model,on,once,pre,show,slot,text,memo"
), Mn = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return ((n) => t[n] || (t[n] = e(n)));
}, ei = /-\w/g, $e = Mn(
  (e) => e.replace(ei, (t) => t.slice(1).toUpperCase())
), ti = /\B([A-Z])/g, ut = Mn(
  (e) => e.replace(ti, "-$1").toLowerCase()
), Rn = Mn((e) => e.charAt(0).toUpperCase() + e.slice(1)), dt = Mn(
  (e) => e ? `on${Rn(e)}` : ""
), lt = (e, t) => !Object.is(e, t), Mt = (e, ...t) => {
  for (let n = 0; n < e.length; n++)
    e[n](...t);
}, On = (e, t, n, o = !1) => {
  Object.defineProperty(e, t, {
    configurable: !0,
    enumerable: !1,
    writable: o,
    value: n
  });
}, ni = (e) => {
  const t = parseFloat(e);
  return isNaN(t) ? e : t;
};
let zo;
const nn = () => zo || (zo = typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : {});
function wo(e) {
  if (T(e)) {
    const t = {};
    for (let n = 0; n < e.length; n++) {
      const o = e[n], s = X(o) ? ii(o) : wo(o);
      if (s)
        for (const r in s)
          t[r] = s[r];
    }
    return t;
  } else if (X(e) || U(e))
    return e;
}
const oi = /;(?![^(]*\))/g, si = /:([^]+)/, ri = /\/\*[^]*?\*\//g;
function ii(e) {
  const t = {};
  return e.replace(ri, "").split(oi).forEach((n) => {
    if (n) {
      const o = n.split(si);
      o.length > 1 && (t[o[0].trim()] = o[1].trim());
    }
  }), t;
}
function Fn(e) {
  let t = "";
  if (X(e))
    t = e;
  else if (T(e))
    for (let n = 0; n < e.length; n++) {
      const o = Fn(e[n]);
      o && (t += o + " ");
    }
  else if (U(e))
    for (const n in e)
      e[n] && (t += n + " ");
  return t.trim();
}
const li = "html,body,base,head,link,meta,style,title,address,article,aside,footer,header,hgroup,h1,h2,h3,h4,h5,h6,nav,section,div,dd,dl,dt,figcaption,figure,picture,hr,img,li,main,ol,p,pre,ul,a,b,abbr,bdi,bdo,br,cite,code,data,dfn,em,i,kbd,mark,q,rp,rt,ruby,s,samp,small,span,strong,sub,sup,time,u,var,wbr,area,audio,map,track,video,embed,object,param,source,canvas,script,noscript,del,ins,caption,col,colgroup,table,thead,tbody,td,th,tr,button,datalist,fieldset,form,input,label,legend,meter,optgroup,option,output,progress,select,textarea,details,dialog,menu,summary,template,blockquote,iframe,tfoot", ci = "svg,animate,animateMotion,animateTransform,circle,clipPath,color-profile,defs,desc,discard,ellipse,feBlend,feColorMatrix,feComponentTransfer,feComposite,feConvolveMatrix,feDiffuseLighting,feDisplacementMap,feDistantLight,feDropShadow,feFlood,feFuncA,feFuncB,feFuncG,feFuncR,feGaussianBlur,feImage,feMerge,feMergeNode,feMorphology,feOffset,fePointLight,feSpecularLighting,feSpotLight,feTile,feTurbulence,filter,foreignObject,g,hatch,hatchpath,image,line,linearGradient,marker,mask,mesh,meshgradient,meshpatch,meshrow,metadata,mpath,path,pattern,polygon,polyline,radialGradient,rect,set,solidcolor,stop,switch,symbol,text,textPath,title,tspan,unknown,use,view", ui = "annotation,annotation-xml,maction,maligngroup,malignmark,math,menclose,merror,mfenced,mfrac,mfraction,mglyph,mi,mlabeledtr,mlongdiv,mmultiscripts,mn,mo,mover,mpadded,mphantom,mprescripts,mroot,mrow,ms,mscarries,mscarry,msgroup,msline,mspace,msqrt,msrow,mstack,mstyle,msub,msubsup,msup,mtable,mtd,mtext,mtr,munder,munderover,none,semantics", fi = /* @__PURE__ */ tt(li), ai = /* @__PURE__ */ tt(ci), pi = /* @__PURE__ */ tt(ui), di = "itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly", hi = /* @__PURE__ */ tt(di);
function js(e) {
  return !!e || e === "";
}
function gi(e, t) {
  if (e.length !== t.length) return !1;
  let n = !0;
  for (let o = 0; n && o < e.length; o++)
    n = Do(e[o], t[o]);
  return n;
}
function Do(e, t) {
  if (e === t) return !0;
  let n = Jo(e), o = Jo(t);
  if (n || o)
    return n && o ? e.getTime() === t.getTime() : !1;
  if (n = qe(e), o = qe(t), n || o)
    return e === t;
  if (n = T(e), o = T(t), n || o)
    return n && o ? gi(e, t) : !1;
  if (n = U(e), o = U(t), n || o) {
    if (!n || !o)
      return !1;
    const s = Object.keys(e).length, r = Object.keys(t).length;
    if (s !== r)
      return !1;
    for (const l in e) {
      const i = e.hasOwnProperty(l), u = t.hasOwnProperty(l);
      if (i && !u || !i && u || !Do(e[l], t[l]))
        return !1;
    }
  }
  return String(e) === String(t);
}
const Hs = (e) => !!(e && e.__v_isRef === !0), dn = (e) => X(e) ? e : e == null ? "" : T(e) || U(e) && (e.toString === Rs || !M(e.toString)) ? Hs(e) ? dn(e.value) : JSON.stringify(e, ks, 2) : String(e), ks = (e, t) => Hs(t) ? ks(e, t.value) : gt(t) ? {
  [`Map(${t.size})`]: [...t.entries()].reduce(
    (n, [o, s], r) => (n[qn(o, r) + " =>"] = s, n),
    {}
  )
} : Ms(t) ? {
  [`Set(${t.size})`]: [...t.values()].map((n) => qn(n))
} : qe(t) ? qn(t) : U(t) && !T(t) && !Fs(t) ? String(t) : t, qn = (e, t = "") => {
  var n;
  return (
    // Symbol.description in es2019+ so we need to cast here to pass
    // the lib: es2016 check
    qe(e) ? `Symbol(${(n = e.description) != null ? n : t})` : e
  );
};
/**
* @vue/reactivity v3.5.29
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/
function Pe(e, ...t) {
  console.warn(`[Vue warn] ${e}`, ...t);
}
let ye;
class vi {
  // TODO isolatedDeclarations "__v_skip"
  constructor(t = !1) {
    this.detached = t, this._active = !0, this._on = 0, this.effects = [], this.cleanups = [], this._isPaused = !1, this.__v_skip = !0, this.parent = ye, !t && ye && (this.index = (ye.scopes || (ye.scopes = [])).push(
      this
    ) - 1);
  }
  get active() {
    return this._active;
  }
  pause() {
    if (this._active) {
      this._isPaused = !0;
      let t, n;
      if (this.scopes)
        for (t = 0, n = this.scopes.length; t < n; t++)
          this.scopes[t].pause();
      for (t = 0, n = this.effects.length; t < n; t++)
        this.effects[t].pause();
    }
  }
  /**
   * Resumes the effect scope, including all child scopes and effects.
   */
  resume() {
    if (this._active && this._isPaused) {
      this._isPaused = !1;
      let t, n;
      if (this.scopes)
        for (t = 0, n = this.scopes.length; t < n; t++)
          this.scopes[t].resume();
      for (t = 0, n = this.effects.length; t < n; t++)
        this.effects[t].resume();
    }
  }
  run(t) {
    if (this._active) {
      const n = ye;
      try {
        return ye = this, t();
      } finally {
        ye = n;
      }
    } else process.env.NODE_ENV !== "production" && Pe("cannot run an inactive effect scope.");
  }
  /**
   * This should only be called on non-detached scopes
   * @internal
   */
  on() {
    ++this._on === 1 && (this.prevScope = ye, ye = this);
  }
  /**
   * This should only be called on non-detached scopes
   * @internal
   */
  off() {
    this._on > 0 && --this._on === 0 && (ye = this.prevScope, this.prevScope = void 0);
  }
  stop(t) {
    if (this._active) {
      this._active = !1;
      let n, o;
      for (n = 0, o = this.effects.length; n < o; n++)
        this.effects[n].stop();
      for (this.effects.length = 0, n = 0, o = this.cleanups.length; n < o; n++)
        this.cleanups[n]();
      if (this.cleanups.length = 0, this.scopes) {
        for (n = 0, o = this.scopes.length; n < o; n++)
          this.scopes[n].stop(!0);
        this.scopes.length = 0;
      }
      if (!this.detached && this.parent && !t) {
        const s = this.parent.scopes.pop();
        s && s !== this && (this.parent.scopes[this.index] = s, s.index = this.index);
      }
      this.parent = void 0;
    }
  }
}
function mi() {
  return ye;
}
let G;
const Yn = /* @__PURE__ */ new WeakSet();
class Ls {
  constructor(t) {
    this.fn = t, this.deps = void 0, this.depsTail = void 0, this.flags = 5, this.next = void 0, this.cleanup = void 0, this.scheduler = void 0, ye && ye.active && ye.effects.push(this);
  }
  pause() {
    this.flags |= 64;
  }
  resume() {
    this.flags & 64 && (this.flags &= -65, Yn.has(this) && (Yn.delete(this), this.trigger()));
  }
  /**
   * @internal
   */
  notify() {
    this.flags & 2 && !(this.flags & 32) || this.flags & 8 || Bs(this);
  }
  run() {
    if (!(this.flags & 1))
      return this.fn();
    this.flags |= 2, Qo(this), Us(this);
    const t = G, n = Ie;
    G = this, Ie = !0;
    try {
      return this.fn();
    } finally {
      process.env.NODE_ENV !== "production" && G !== this && Pe(
        "Active effect was not restored correctly - this is likely a Vue internal bug."
      ), Ks(this), G = t, Ie = n, this.flags &= -3;
    }
  }
  stop() {
    if (this.flags & 1) {
      for (let t = this.deps; t; t = t.nextDep)
        Co(t);
      this.deps = this.depsTail = void 0, Qo(this), this.onStop && this.onStop(), this.flags &= -2;
    }
  }
  trigger() {
    this.flags & 64 ? Yn.add(this) : this.scheduler ? this.scheduler() : this.runIfDirty();
  }
  /**
   * @internal
   */
  runIfDirty() {
    ro(this) && this.run();
  }
  get dirty() {
    return ro(this);
  }
}
let Ws = 0, Ut, Kt;
function Bs(e, t = !1) {
  if (e.flags |= 8, t) {
    e.next = Kt, Kt = e;
    return;
  }
  e.next = Ut, Ut = e;
}
function Vo() {
  Ws++;
}
function So() {
  if (--Ws > 0)
    return;
  if (Kt) {
    let t = Kt;
    for (Kt = void 0; t; ) {
      const n = t.next;
      t.next = void 0, t.flags &= -9, t = n;
    }
  }
  let e;
  for (; Ut; ) {
    let t = Ut;
    for (Ut = void 0; t; ) {
      const n = t.next;
      if (t.next = void 0, t.flags &= -9, t.flags & 1)
        try {
          t.trigger();
        } catch (o) {
          e || (e = o);
        }
      t = n;
    }
  }
  if (e) throw e;
}
function Us(e) {
  for (let t = e.deps; t; t = t.nextDep)
    t.version = -1, t.prevActiveLink = t.dep.activeLink, t.dep.activeLink = t;
}
function Ks(e) {
  let t, n = e.depsTail, o = n;
  for (; o; ) {
    const s = o.prevDep;
    o.version === -1 ? (o === n && (n = s), Co(o), _i(o)) : t = o, o.dep.activeLink = o.prevActiveLink, o.prevActiveLink = void 0, o = s;
  }
  e.deps = t, e.depsTail = n;
}
function ro(e) {
  for (let t = e.deps; t; t = t.nextDep)
    if (t.dep.version !== t.version || t.dep.computed && (Gs(t.dep.computed) || t.dep.version !== t.version))
      return !0;
  return !!e._dirty;
}
function Gs(e) {
  if (e.flags & 4 && !(e.flags & 16) || (e.flags &= -17, e.globalVersion === Jt) || (e.globalVersion = Jt, !e.isSSR && e.flags & 128 && (!e.deps && !e._dirty || !ro(e))))
    return;
  e.flags |= 2;
  const t = e.dep, n = G, o = Ie;
  G = e, Ie = !0;
  try {
    Us(e);
    const s = e.fn(e._value);
    (t.version === 0 || lt(s, e._value)) && (e.flags |= 128, e._value = s, t.version++);
  } catch (s) {
    throw t.version++, s;
  } finally {
    G = n, Ie = o, Ks(e), e.flags &= -3;
  }
}
function Co(e, t = !1) {
  const { dep: n, prevSub: o, nextSub: s } = e;
  if (o && (o.nextSub = s, e.prevSub = void 0), s && (s.prevSub = o, e.nextSub = void 0), process.env.NODE_ENV !== "production" && n.subsHead === e && (n.subsHead = s), n.subs === e && (n.subs = o, !o && n.computed)) {
    n.computed.flags &= -5;
    for (let r = n.computed.deps; r; r = r.nextDep)
      Co(r, !0);
  }
  !t && !--n.sc && n.map && n.map.delete(n.key);
}
function _i(e) {
  const { prevDep: t, nextDep: n } = e;
  t && (t.nextDep = n, e.prevDep = void 0), n && (n.prevDep = t, e.nextDep = void 0);
}
let Ie = !0;
const qs = [];
function Ae() {
  qs.push(Ie), Ie = !1;
}
function Me() {
  const e = qs.pop();
  Ie = e === void 0 ? !0 : e;
}
function Qo(e) {
  const { cleanup: t } = e;
  if (e.cleanup = void 0, t) {
    const n = G;
    G = void 0;
    try {
      t();
    } finally {
      G = n;
    }
  }
}
let Jt = 0;
class Ei {
  constructor(t, n) {
    this.sub = t, this.dep = n, this.version = n.version, this.nextDep = this.prevDep = this.nextSub = this.prevSub = this.prevActiveLink = void 0;
  }
}
class To {
  // TODO isolatedDeclarations "__v_skip"
  constructor(t) {
    this.computed = t, this.version = 0, this.activeLink = void 0, this.subs = void 0, this.map = void 0, this.key = void 0, this.sc = 0, this.__v_skip = !0, process.env.NODE_ENV !== "production" && (this.subsHead = void 0);
  }
  track(t) {
    if (!G || !Ie || G === this.computed)
      return;
    let n = this.activeLink;
    if (n === void 0 || n.sub !== G)
      n = this.activeLink = new Ei(G, this), G.deps ? (n.prevDep = G.depsTail, G.depsTail.nextDep = n, G.depsTail = n) : G.deps = G.depsTail = n, Ys(n);
    else if (n.version === -1 && (n.version = this.version, n.nextDep)) {
      const o = n.nextDep;
      o.prevDep = n.prevDep, n.prevDep && (n.prevDep.nextDep = o), n.prevDep = G.depsTail, n.nextDep = void 0, G.depsTail.nextDep = n, G.depsTail = n, G.deps === n && (G.deps = o);
    }
    return process.env.NODE_ENV !== "production" && G.onTrack && G.onTrack(
      Z(
        {
          effect: G
        },
        t
      )
    ), n;
  }
  trigger(t) {
    this.version++, Jt++, this.notify(t);
  }
  notify(t) {
    Vo();
    try {
      if (process.env.NODE_ENV !== "production")
        for (let n = this.subsHead; n; n = n.nextSub)
          n.sub.onTrigger && !(n.sub.flags & 8) && n.sub.onTrigger(
            Z(
              {
                effect: n.sub
              },
              t
            )
          );
      for (let n = this.subs; n; n = n.prevSub)
        n.sub.notify() && n.sub.dep.notify();
    } finally {
      So();
    }
  }
}
function Ys(e) {
  if (e.dep.sc++, e.sub.flags & 4) {
    const t = e.dep.computed;
    if (t && !e.dep.subs) {
      t.flags |= 20;
      for (let o = t.deps; o; o = o.nextDep)
        Ys(o);
    }
    const n = e.dep.subs;
    n !== e && (e.prevSub = n, n && (n.nextSub = e)), process.env.NODE_ENV !== "production" && e.dep.subsHead === void 0 && (e.dep.subsHead = e), e.dep.subs = e;
  }
}
const io = /* @__PURE__ */ new WeakMap(), vt = /* @__PURE__ */ Symbol(
  process.env.NODE_ENV !== "production" ? "Object iterate" : ""
), lo = /* @__PURE__ */ Symbol(
  process.env.NODE_ENV !== "production" ? "Map keys iterate" : ""
), zt = /* @__PURE__ */ Symbol(
  process.env.NODE_ENV !== "production" ? "Array iterate" : ""
);
function re(e, t, n) {
  if (Ie && G) {
    let o = io.get(e);
    o || io.set(e, o = /* @__PURE__ */ new Map());
    let s = o.get(n);
    s || (o.set(n, s = new To()), s.map = o, s.key = n), process.env.NODE_ENV !== "production" ? s.track({
      target: e,
      type: t,
      key: n
    }) : s.track();
  }
}
function Ue(e, t, n, o, s, r) {
  const l = io.get(e);
  if (!l) {
    Jt++;
    return;
  }
  const i = (u) => {
    u && (process.env.NODE_ENV !== "production" ? u.trigger({
      target: e,
      type: t,
      key: n,
      newValue: o,
      oldValue: s,
      oldTarget: r
    }) : u.trigger());
  };
  if (Vo(), t === "clear")
    l.forEach(i);
  else {
    const u = T(e), d = u && xo(n);
    if (u && n === "length") {
      const a = Number(o);
      l.forEach((p, g) => {
        (g === "length" || g === zt || !qe(g) && g >= a) && i(p);
      });
    } else
      switch ((n !== void 0 || l.has(void 0)) && i(l.get(n)), d && i(l.get(zt)), t) {
        case "add":
          u ? d && i(l.get("length")) : (i(l.get(vt)), gt(e) && i(l.get(lo)));
          break;
        case "delete":
          u || (i(l.get(vt)), gt(e) && i(l.get(lo)));
          break;
        case "set":
          gt(e) && i(l.get(vt));
          break;
      }
  }
  So();
}
function Nt(e) {
  const t = /* @__PURE__ */ j(e);
  return t === e ? t : (re(t, "iterate", zt), /* @__PURE__ */ he(e) ? t : t.map(Fe));
}
function jn(e) {
  return re(e = /* @__PURE__ */ j(e), "iterate", zt), e;
}
function st(e, t) {
  return /* @__PURE__ */ Re(e) ? St(/* @__PURE__ */ ct(e) ? Fe(t) : t) : Fe(t);
}
const yi = {
  __proto__: null,
  [Symbol.iterator]() {
    return Jn(this, Symbol.iterator, (e) => st(this, e));
  },
  concat(...e) {
    return Nt(this).concat(
      ...e.map((t) => T(t) ? Nt(t) : t)
    );
  },
  entries() {
    return Jn(this, "entries", (e) => (e[1] = st(this, e[1]), e));
  },
  every(e, t) {
    return Je(this, "every", e, t, void 0, arguments);
  },
  filter(e, t) {
    return Je(
      this,
      "filter",
      e,
      t,
      (n) => n.map((o) => st(this, o)),
      arguments
    );
  },
  find(e, t) {
    return Je(
      this,
      "find",
      e,
      t,
      (n) => st(this, n),
      arguments
    );
  },
  findIndex(e, t) {
    return Je(this, "findIndex", e, t, void 0, arguments);
  },
  findLast(e, t) {
    return Je(
      this,
      "findLast",
      e,
      t,
      (n) => st(this, n),
      arguments
    );
  },
  findLastIndex(e, t) {
    return Je(this, "findLastIndex", e, t, void 0, arguments);
  },
  // flat, flatMap could benefit from ARRAY_ITERATE but are not straight-forward to implement
  forEach(e, t) {
    return Je(this, "forEach", e, t, void 0, arguments);
  },
  includes(...e) {
    return zn(this, "includes", e);
  },
  indexOf(...e) {
    return zn(this, "indexOf", e);
  },
  join(e) {
    return Nt(this).join(e);
  },
  // keys() iterator only reads `length`, no optimization required
  lastIndexOf(...e) {
    return zn(this, "lastIndexOf", e);
  },
  map(e, t) {
    return Je(this, "map", e, t, void 0, arguments);
  },
  pop() {
    return Rt(this, "pop");
  },
  push(...e) {
    return Rt(this, "push", e);
  },
  reduce(e, ...t) {
    return Xo(this, "reduce", e, t);
  },
  reduceRight(e, ...t) {
    return Xo(this, "reduceRight", e, t);
  },
  shift() {
    return Rt(this, "shift");
  },
  // slice could use ARRAY_ITERATE but also seems to beg for range tracking
  some(e, t) {
    return Je(this, "some", e, t, void 0, arguments);
  },
  splice(...e) {
    return Rt(this, "splice", e);
  },
  toReversed() {
    return Nt(this).toReversed();
  },
  toSorted(e) {
    return Nt(this).toSorted(e);
  },
  toSpliced(...e) {
    return Nt(this).toSpliced(...e);
  },
  unshift(...e) {
    return Rt(this, "unshift", e);
  },
  values() {
    return Jn(this, "values", (e) => st(this, e));
  }
};
function Jn(e, t, n) {
  const o = jn(e), s = o[t]();
  return o !== e && !/* @__PURE__ */ he(e) && (s._next = s.next, s.next = () => {
    const r = s._next();
    return r.done || (r.value = n(r.value)), r;
  }), s;
}
const Ni = Array.prototype;
function Je(e, t, n, o, s, r) {
  const l = jn(e), i = l !== e && !/* @__PURE__ */ he(e), u = l[t];
  if (u !== Ni[t]) {
    const p = u.apply(e, r);
    return i ? Fe(p) : p;
  }
  let d = n;
  l !== e && (i ? d = function(p, g) {
    return n.call(this, st(e, p), g, e);
  } : n.length > 2 && (d = function(p, g) {
    return n.call(this, p, g, e);
  }));
  const a = u.call(l, d, o);
  return i && s ? s(a) : a;
}
function Xo(e, t, n, o) {
  const s = jn(e);
  let r = n;
  return s !== e && (/* @__PURE__ */ he(e) ? n.length > 3 && (r = function(l, i, u) {
    return n.call(this, l, i, u, e);
  }) : r = function(l, i, u) {
    return n.call(this, l, st(e, i), u, e);
  }), s[t](r, ...o);
}
function zn(e, t, n) {
  const o = /* @__PURE__ */ j(e);
  re(o, "iterate", zt);
  const s = o[t](...n);
  return (s === -1 || s === !1) && /* @__PURE__ */ xn(n[0]) ? (n[0] = /* @__PURE__ */ j(n[0]), o[t](...n)) : s;
}
function Rt(e, t, n = []) {
  Ae(), Vo();
  const o = (/* @__PURE__ */ j(e))[t].apply(e, n);
  return So(), Me(), o;
}
const bi = /* @__PURE__ */ tt("__proto__,__v_isRef,__isVue"), Js = new Set(
  /* @__PURE__ */ Object.getOwnPropertyNames(Symbol).filter((e) => e !== "arguments" && e !== "caller").map((e) => Symbol[e]).filter(qe)
);
function Oi(e) {
  qe(e) || (e = String(e));
  const t = /* @__PURE__ */ j(this);
  return re(t, "has", e), t.hasOwnProperty(e);
}
class zs {
  constructor(t = !1, n = !1) {
    this._isReadonly = t, this._isShallow = n;
  }
  get(t, n, o) {
    if (n === "__v_skip") return t.__v_skip;
    const s = this._isReadonly, r = this._isShallow;
    if (n === "__v_isReactive")
      return !s;
    if (n === "__v_isReadonly")
      return s;
    if (n === "__v_isShallow")
      return r;
    if (n === "__v_raw")
      return o === (s ? r ? nr : tr : r ? er : Zs).get(t) || // receiver is not the reactive proxy, but has the same prototype
      // this means the receiver is a user proxy of the reactive proxy
      Object.getPrototypeOf(t) === Object.getPrototypeOf(o) ? t : void 0;
    const l = T(t);
    if (!s) {
      let u;
      if (l && (u = yi[n]))
        return u;
      if (n === "hasOwnProperty")
        return Oi;
    }
    const i = Reflect.get(
      t,
      n,
      // if this is a proxy wrapping a ref, return methods using the raw ref
      // as receiver so that we don't have to call `toRaw` on the ref in all
      // its class methods
      /* @__PURE__ */ ne(t) ? t : o
    );
    if ((qe(n) ? Js.has(n) : bi(n)) || (s || re(t, "get", n), r))
      return i;
    if (/* @__PURE__ */ ne(i)) {
      const u = l && xo(n) ? i : i.value;
      return s && U(u) ? /* @__PURE__ */ uo(u) : u;
    }
    return U(i) ? s ? /* @__PURE__ */ uo(i) : /* @__PURE__ */ $o(i) : i;
  }
}
class Qs extends zs {
  constructor(t = !1) {
    super(!1, t);
  }
  set(t, n, o, s) {
    let r = t[n];
    const l = T(t) && xo(n);
    if (!this._isShallow) {
      const d = /* @__PURE__ */ Re(r);
      if (!/* @__PURE__ */ he(o) && !/* @__PURE__ */ Re(o) && (r = /* @__PURE__ */ j(r), o = /* @__PURE__ */ j(o)), !l && /* @__PURE__ */ ne(r) && !/* @__PURE__ */ ne(o))
        return d ? (process.env.NODE_ENV !== "production" && Pe(
          `Set operation on key "${String(n)}" failed: target is readonly.`,
          t[n]
        ), !0) : (r.value = o, !0);
    }
    const i = l ? Number(n) < t.length : B(t, n), u = Reflect.set(
      t,
      n,
      o,
      /* @__PURE__ */ ne(t) ? t : s
    );
    return t === /* @__PURE__ */ j(s) && (i ? lt(o, r) && Ue(t, "set", n, o, r) : Ue(t, "add", n, o)), u;
  }
  deleteProperty(t, n) {
    const o = B(t, n), s = t[n], r = Reflect.deleteProperty(t, n);
    return r && o && Ue(t, "delete", n, void 0, s), r;
  }
  has(t, n) {
    const o = Reflect.has(t, n);
    return (!qe(n) || !Js.has(n)) && re(t, "has", n), o;
  }
  ownKeys(t) {
    return re(
      t,
      "iterate",
      T(t) ? "length" : vt
    ), Reflect.ownKeys(t);
  }
}
class Xs extends zs {
  constructor(t = !1) {
    super(!0, t);
  }
  set(t, n) {
    return process.env.NODE_ENV !== "production" && Pe(
      `Set operation on key "${String(n)}" failed: target is readonly.`,
      t
    ), !0;
  }
  deleteProperty(t, n) {
    return process.env.NODE_ENV !== "production" && Pe(
      `Delete operation on key "${String(n)}" failed: target is readonly.`,
      t
    ), !0;
  }
}
const xi = /* @__PURE__ */ new Qs(), wi = /* @__PURE__ */ new Xs(), Di = /* @__PURE__ */ new Qs(!0), Vi = /* @__PURE__ */ new Xs(!0), co = (e) => e, fn = (e) => Reflect.getPrototypeOf(e);
function Si(e, t, n) {
  return function(...o) {
    const s = this.__v_raw, r = /* @__PURE__ */ j(s), l = gt(r), i = e === "entries" || e === Symbol.iterator && l, u = e === "keys" && l, d = s[e](...o), a = n ? co : t ? St : Fe;
    return !t && re(
      r,
      "iterate",
      u ? lo : vt
    ), Z(
      // inheriting all iterator properties
      Object.create(d),
      {
        // iterator protocol
        next() {
          const { value: p, done: g } = d.next();
          return g ? { value: p, done: g } : {
            value: i ? [a(p[0]), a(p[1])] : a(p),
            done: g
          };
        }
      }
    );
  };
}
function an(e) {
  return function(...t) {
    if (process.env.NODE_ENV !== "production") {
      const n = t[0] ? `on key "${t[0]}" ` : "";
      Pe(
        `${Rn(e)} operation ${n}failed: target is readonly.`,
        /* @__PURE__ */ j(this)
      );
    }
    return e === "delete" ? !1 : e === "clear" ? void 0 : this;
  };
}
function Ci(e, t) {
  const n = {
    get(s) {
      const r = this.__v_raw, l = /* @__PURE__ */ j(r), i = /* @__PURE__ */ j(s);
      e || (lt(s, i) && re(l, "get", s), re(l, "get", i));
      const { has: u } = fn(l), d = t ? co : e ? St : Fe;
      if (u.call(l, s))
        return d(r.get(s));
      if (u.call(l, i))
        return d(r.get(i));
      r !== l && r.get(s);
    },
    get size() {
      const s = this.__v_raw;
      return !e && re(/* @__PURE__ */ j(s), "iterate", vt), s.size;
    },
    has(s) {
      const r = this.__v_raw, l = /* @__PURE__ */ j(r), i = /* @__PURE__ */ j(s);
      return e || (lt(s, i) && re(l, "has", s), re(l, "has", i)), s === i ? r.has(s) : r.has(s) || r.has(i);
    },
    forEach(s, r) {
      const l = this, i = l.__v_raw, u = /* @__PURE__ */ j(i), d = t ? co : e ? St : Fe;
      return !e && re(u, "iterate", vt), i.forEach((a, p) => s.call(r, d(a), d(p), l));
    }
  };
  return Z(
    n,
    e ? {
      add: an("add"),
      set: an("set"),
      delete: an("delete"),
      clear: an("clear")
    } : {
      add(s) {
        !t && !/* @__PURE__ */ he(s) && !/* @__PURE__ */ Re(s) && (s = /* @__PURE__ */ j(s));
        const r = /* @__PURE__ */ j(this);
        return fn(r).has.call(r, s) || (r.add(s), Ue(r, "add", s, s)), this;
      },
      set(s, r) {
        !t && !/* @__PURE__ */ he(r) && !/* @__PURE__ */ Re(r) && (r = /* @__PURE__ */ j(r));
        const l = /* @__PURE__ */ j(this), { has: i, get: u } = fn(l);
        let d = i.call(l, s);
        d ? process.env.NODE_ENV !== "production" && Zo(l, i, s) : (s = /* @__PURE__ */ j(s), d = i.call(l, s));
        const a = u.call(l, s);
        return l.set(s, r), d ? lt(r, a) && Ue(l, "set", s, r, a) : Ue(l, "add", s, r), this;
      },
      delete(s) {
        const r = /* @__PURE__ */ j(this), { has: l, get: i } = fn(r);
        let u = l.call(r, s);
        u ? process.env.NODE_ENV !== "production" && Zo(r, l, s) : (s = /* @__PURE__ */ j(s), u = l.call(r, s));
        const d = i ? i.call(r, s) : void 0, a = r.delete(s);
        return u && Ue(r, "delete", s, void 0, d), a;
      },
      clear() {
        const s = /* @__PURE__ */ j(this), r = s.size !== 0, l = process.env.NODE_ENV !== "production" ? gt(s) ? new Map(s) : new Set(s) : void 0, i = s.clear();
        return r && Ue(
          s,
          "clear",
          void 0,
          void 0,
          l
        ), i;
      }
    }
  ), [
    "keys",
    "values",
    "entries",
    Symbol.iterator
  ].forEach((s) => {
    n[s] = Si(s, e, t);
  }), n;
}
function Hn(e, t) {
  const n = Ci(e, t);
  return (o, s, r) => s === "__v_isReactive" ? !e : s === "__v_isReadonly" ? e : s === "__v_raw" ? o : Reflect.get(
    B(n, s) && s in o ? n : o,
    s,
    r
  );
}
const Ti = {
  get: /* @__PURE__ */ Hn(!1, !1)
}, $i = {
  get: /* @__PURE__ */ Hn(!1, !0)
}, Ii = {
  get: /* @__PURE__ */ Hn(!0, !1)
}, Pi = {
  get: /* @__PURE__ */ Hn(!0, !0)
};
function Zo(e, t, n) {
  const o = /* @__PURE__ */ j(n);
  if (o !== n && t.call(e, o)) {
    const s = Oo(e);
    Pe(
      `Reactive ${s} contains both the raw and reactive versions of the same object${s === "Map" ? " as keys" : ""}, which can lead to inconsistencies. Avoid differentiating between the raw and reactive versions of an object and only use the reactive version if possible.`
    );
  }
}
const Zs = /* @__PURE__ */ new WeakMap(), er = /* @__PURE__ */ new WeakMap(), tr = /* @__PURE__ */ new WeakMap(), nr = /* @__PURE__ */ new WeakMap();
function Ai(e) {
  switch (e) {
    case "Object":
    case "Array":
      return 1;
    case "Map":
    case "Set":
    case "WeakMap":
    case "WeakSet":
      return 2;
    default:
      return 0;
  }
}
function Mi(e) {
  return e.__v_skip || !Object.isExtensible(e) ? 0 : Ai(Oo(e));
}
// @__NO_SIDE_EFFECTS__
function $o(e) {
  return /* @__PURE__ */ Re(e) ? e : kn(
    e,
    !1,
    xi,
    Ti,
    Zs
  );
}
// @__NO_SIDE_EFFECTS__
function Ri(e) {
  return kn(
    e,
    !1,
    Di,
    $i,
    er
  );
}
// @__NO_SIDE_EFFECTS__
function uo(e) {
  return kn(
    e,
    !0,
    wi,
    Ii,
    tr
  );
}
// @__NO_SIDE_EFFECTS__
function Ke(e) {
  return kn(
    e,
    !0,
    Vi,
    Pi,
    nr
  );
}
function kn(e, t, n, o, s) {
  if (!U(e))
    return process.env.NODE_ENV !== "production" && Pe(
      `value cannot be made ${t ? "readonly" : "reactive"}: ${String(
        e
      )}`
    ), e;
  if (e.__v_raw && !(t && e.__v_isReactive))
    return e;
  const r = Mi(e);
  if (r === 0)
    return e;
  const l = s.get(e);
  if (l)
    return l;
  const i = new Proxy(
    e,
    r === 2 ? o : n
  );
  return s.set(e, i), i;
}
// @__NO_SIDE_EFFECTS__
function ct(e) {
  return /* @__PURE__ */ Re(e) ? /* @__PURE__ */ ct(e.__v_raw) : !!(e && e.__v_isReactive);
}
// @__NO_SIDE_EFFECTS__
function Re(e) {
  return !!(e && e.__v_isReadonly);
}
// @__NO_SIDE_EFFECTS__
function he(e) {
  return !!(e && e.__v_isShallow);
}
// @__NO_SIDE_EFFECTS__
function xn(e) {
  return e ? !!e.__v_raw : !1;
}
// @__NO_SIDE_EFFECTS__
function j(e) {
  const t = e && e.__v_raw;
  return t ? /* @__PURE__ */ j(t) : e;
}
function Fi(e) {
  return !B(e, "__v_skip") && Object.isExtensible(e) && On(e, "__v_skip", !0), e;
}
const Fe = (e) => U(e) ? /* @__PURE__ */ $o(e) : e, St = (e) => U(e) ? /* @__PURE__ */ uo(e) : e;
// @__NO_SIDE_EFFECTS__
function ne(e) {
  return e ? e.__v_isRef === !0 : !1;
}
// @__NO_SIDE_EFFECTS__
function Ft(e) {
  return ji(e, !1);
}
function ji(e, t) {
  return /* @__PURE__ */ ne(e) ? e : new Hi(e, t);
}
class Hi {
  constructor(t, n) {
    this.dep = new To(), this.__v_isRef = !0, this.__v_isShallow = !1, this._rawValue = n ? t : /* @__PURE__ */ j(t), this._value = n ? t : Fe(t), this.__v_isShallow = n;
  }
  get value() {
    return process.env.NODE_ENV !== "production" ? this.dep.track({
      target: this,
      type: "get",
      key: "value"
    }) : this.dep.track(), this._value;
  }
  set value(t) {
    const n = this._rawValue, o = this.__v_isShallow || /* @__PURE__ */ he(t) || /* @__PURE__ */ Re(t);
    t = o ? t : /* @__PURE__ */ j(t), lt(t, n) && (this._rawValue = t, this._value = o ? t : Fe(t), process.env.NODE_ENV !== "production" ? this.dep.trigger({
      target: this,
      type: "set",
      key: "value",
      newValue: t,
      oldValue: n
    }) : this.dep.trigger());
  }
}
function or(e) {
  return /* @__PURE__ */ ne(e) ? e.value : e;
}
const ki = {
  get: (e, t, n) => t === "__v_raw" ? e : or(Reflect.get(e, t, n)),
  set: (e, t, n, o) => {
    const s = e[t];
    return /* @__PURE__ */ ne(s) && !/* @__PURE__ */ ne(n) ? (s.value = n, !0) : Reflect.set(e, t, n, o);
  }
};
function sr(e) {
  return /* @__PURE__ */ ct(e) ? e : new Proxy(e, ki);
}
class Li {
  constructor(t, n, o) {
    this.fn = t, this.setter = n, this._value = void 0, this.dep = new To(this), this.__v_isRef = !0, this.deps = void 0, this.depsTail = void 0, this.flags = 16, this.globalVersion = Jt - 1, this.next = void 0, this.effect = this, this.__v_isReadonly = !n, this.isSSR = o;
  }
  /**
   * @internal
   */
  notify() {
    if (this.flags |= 16, !(this.flags & 8) && // avoid infinite self recursion
    G !== this)
      return Bs(this, !0), !0;
    process.env.NODE_ENV;
  }
  get value() {
    const t = process.env.NODE_ENV !== "production" ? this.dep.track({
      target: this,
      type: "get",
      key: "value"
    }) : this.dep.track();
    return Gs(this), t && (t.version = this.dep.version), this._value;
  }
  set value(t) {
    this.setter ? this.setter(t) : process.env.NODE_ENV !== "production" && Pe("Write operation failed: computed value is readonly");
  }
}
// @__NO_SIDE_EFFECTS__
function Wi(e, t, n = !1) {
  let o, s;
  M(e) ? o = e : (o = e.get, s = e.set);
  const r = new Li(o, s, n);
  return process.env.NODE_ENV, r;
}
const pn = {}, wn = /* @__PURE__ */ new WeakMap();
let ht;
function Bi(e, t = !1, n = ht) {
  if (n) {
    let o = wn.get(n);
    o || wn.set(n, o = []), o.push(e);
  } else process.env.NODE_ENV !== "production" && !t && Pe(
    "onWatcherCleanup() was called when there was no active watcher to associate with."
  );
}
function Ui(e, t, n = Y) {
  const { immediate: o, deep: s, once: r, scheduler: l, augmentJob: i, call: u } = n, d = (C) => {
    (n.onWarn || Pe)(
      "Invalid watch source: ",
      C,
      "A watch source can only be a getter/effect function, a ref, a reactive object, or an array of these types."
    );
  }, a = (C) => s ? C : /* @__PURE__ */ he(C) || s === !1 || s === 0 ? it(C, 1) : it(C);
  let p, g, O, $, D = !1, Q = !1;
  if (/* @__PURE__ */ ne(e) ? (g = () => e.value, D = /* @__PURE__ */ he(e)) : /* @__PURE__ */ ct(e) ? (g = () => a(e), D = !0) : T(e) ? (Q = !0, D = e.some((C) => /* @__PURE__ */ ct(C) || /* @__PURE__ */ he(C)), g = () => e.map((C) => {
    if (/* @__PURE__ */ ne(C))
      return C.value;
    if (/* @__PURE__ */ ct(C))
      return a(C);
    if (M(C))
      return u ? u(C, 2) : C();
    process.env.NODE_ENV !== "production" && d(C);
  })) : M(e) ? t ? g = u ? () => u(e, 2) : e : g = () => {
    if (O) {
      Ae();
      try {
        O();
      } finally {
        Me();
      }
    }
    const C = ht;
    ht = p;
    try {
      return u ? u(e, 3, [$]) : e($);
    } finally {
      ht = C;
    }
  } : (g = ie, process.env.NODE_ENV !== "production" && d(e)), t && s) {
    const C = g, ee = s === !0 ? 1 / 0 : s;
    g = () => it(C(), ee);
  }
  const J = mi(), K = () => {
    p.stop(), J && J.active && No(J.effects, p);
  };
  if (r && t) {
    const C = t;
    t = (...ee) => {
      C(...ee), K();
    };
  }
  let L = Q ? new Array(e.length).fill(pn) : pn;
  const ue = (C) => {
    if (!(!(p.flags & 1) || !p.dirty && !C))
      if (t) {
        const ee = p.run();
        if (s || D || (Q ? ee.some((fe, le) => lt(fe, L[le])) : lt(ee, L))) {
          O && O();
          const fe = ht;
          ht = p;
          try {
            const le = [
              ee,
              // pass undefined as the old value when it's changed for the first time
              L === pn ? void 0 : Q && L[0] === pn ? [] : L,
              $
            ];
            L = ee, u ? u(t, 3, le) : (
              // @ts-expect-error
              t(...le)
            );
          } finally {
            ht = fe;
          }
        }
      } else
        p.run();
  };
  return i && i(ue), p = new Ls(g), p.scheduler = l ? () => l(ue, !1) : ue, $ = (C) => Bi(C, !1, p), O = p.onStop = () => {
    const C = wn.get(p);
    if (C) {
      if (u)
        u(C, 4);
      else
        for (const ee of C) ee();
      wn.delete(p);
    }
  }, process.env.NODE_ENV !== "production" && (p.onTrack = n.onTrack, p.onTrigger = n.onTrigger), t ? o ? ue(!0) : L = p.run() : l ? l(ue.bind(null, !0), !0) : p.run(), K.pause = p.pause.bind(p), K.resume = p.resume.bind(p), K.stop = K, K;
}
function it(e, t = 1 / 0, n) {
  if (t <= 0 || !U(e) || e.__v_skip || (n = n || /* @__PURE__ */ new Map(), (n.get(e) || 0) >= t))
    return e;
  if (n.set(e, t), t--, /* @__PURE__ */ ne(e))
    it(e.value, t, n);
  else if (T(e))
    for (let o = 0; o < e.length; o++)
      it(e[o], t, n);
  else if (Ms(e) || gt(e))
    e.forEach((o) => {
      it(o, t, n);
    });
  else if (Fs(e)) {
    for (const o in e)
      it(e[o], t, n);
    for (const o of Object.getOwnPropertySymbols(e))
      Object.prototype.propertyIsEnumerable.call(e, o) && it(e[o], t, n);
  }
  return e;
}
/**
* @vue/runtime-core v3.5.29
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/
const mt = [];
function hn(e) {
  mt.push(e);
}
function gn() {
  mt.pop();
}
let Qn = !1;
function b(e, ...t) {
  if (Qn) return;
  Qn = !0, Ae();
  const n = mt.length ? mt[mt.length - 1].component : null, o = n && n.appContext.config.warnHandler, s = Ki();
  if (o)
    Tt(
      o,
      n,
      11,
      [
        // eslint-disable-next-line no-restricted-syntax
        e + t.map((r) => {
          var l, i;
          return (i = (l = r.toString) == null ? void 0 : l.call(r)) != null ? i : JSON.stringify(r);
        }).join(""),
        n && n.proxy,
        s.map(
          ({ vnode: r }) => `at <${cn(n, r.type)}>`
        ).join(`
`),
        s
      ]
    );
  else {
    const r = [`[Vue warn]: ${e}`, ...t];
    s.length && r.push(`
`, ...Gi(s)), console.warn(...r);
  }
  Me(), Qn = !1;
}
function Ki() {
  let e = mt[mt.length - 1];
  if (!e)
    return [];
  const t = [];
  for (; e; ) {
    const n = t[0];
    n && n.vnode === e ? n.recurseCount++ : t.push({
      vnode: e,
      recurseCount: 0
    });
    const o = e.component && e.component.parent;
    e = o && o.vnode;
  }
  return t;
}
function Gi(e) {
  const t = [];
  return e.forEach((n, o) => {
    t.push(...o === 0 ? [] : [`
`], ...qi(n));
  }), t;
}
function qi({ vnode: e, recurseCount: t }) {
  const n = t > 0 ? `... (${t} recursive calls)` : "", o = e.component ? e.component.parent == null : !1, s = ` at <${cn(
    e.component,
    e.type,
    o
  )}`, r = ">" + n;
  return e.props ? [s, ...Yi(e.props), r] : [s + r];
}
function Yi(e) {
  const t = [], n = Object.keys(e);
  return n.slice(0, 3).forEach((o) => {
    t.push(...rr(o, e[o]));
  }), n.length > 3 && t.push(" ..."), t;
}
function rr(e, t, n) {
  return X(t) ? (t = JSON.stringify(t), n ? t : [`${e}=${t}`]) : typeof t == "number" || typeof t == "boolean" || t == null ? n ? t : [`${e}=${t}`] : /* @__PURE__ */ ne(t) ? (t = rr(e, /* @__PURE__ */ j(t.value), !0), n ? t : [`${e}=Ref<`, t, ">"]) : M(t) ? [`${e}=fn${t.name ? `<${t.name}>` : ""}`] : (t = /* @__PURE__ */ j(t), n ? t : [`${e}=`, t]);
}
const Io = {
  sp: "serverPrefetch hook",
  bc: "beforeCreate hook",
  c: "created hook",
  bm: "beforeMount hook",
  m: "mounted hook",
  bu: "beforeUpdate hook",
  u: "updated",
  bum: "beforeUnmount hook",
  um: "unmounted hook",
  a: "activated hook",
  da: "deactivated hook",
  ec: "errorCaptured hook",
  rtc: "renderTracked hook",
  rtg: "renderTriggered hook",
  0: "setup function",
  1: "render function",
  2: "watcher getter",
  3: "watcher callback",
  4: "watcher cleanup function",
  5: "native event handler",
  6: "component event handler",
  7: "vnode hook",
  8: "directive hook",
  9: "transition hook",
  10: "app errorHandler",
  11: "app warnHandler",
  12: "ref function",
  13: "async component loader",
  14: "scheduler flush",
  15: "component update",
  16: "app unmount cleanup function"
};
function Tt(e, t, n, o) {
  try {
    return o ? e(...o) : e();
  } catch (s) {
    on(s, t, n);
  }
}
function Ye(e, t, n, o) {
  if (M(e)) {
    const s = Tt(e, t, n, o);
    return s && bo(s) && s.catch((r) => {
      on(r, t, n);
    }), s;
  }
  if (T(e)) {
    const s = [];
    for (let r = 0; r < e.length; r++)
      s.push(Ye(e[r], t, n, o));
    return s;
  } else process.env.NODE_ENV !== "production" && b(
    `Invalid value type passed to callWithAsyncErrorHandling(): ${typeof e}`
  );
}
function on(e, t, n, o = !0) {
  const s = t ? t.vnode : null, { errorHandler: r, throwUnhandledErrorInProduction: l } = t && t.appContext.config || Y;
  if (t) {
    let i = t.parent;
    const u = t.proxy, d = process.env.NODE_ENV !== "production" ? Io[n] : `https://vuejs.org/error-reference/#runtime-${n}`;
    for (; i; ) {
      const a = i.ec;
      if (a) {
        for (let p = 0; p < a.length; p++)
          if (a[p](e, u, d) === !1)
            return;
      }
      i = i.parent;
    }
    if (r) {
      Ae(), Tt(r, null, 10, [
        e,
        u,
        d
      ]), Me();
      return;
    }
  }
  Ji(e, n, s, o, l);
}
function Ji(e, t, n, o = !0, s = !1) {
  if (process.env.NODE_ENV !== "production") {
    const r = Io[t];
    if (n && hn(n), b(`Unhandled error${r ? ` during execution of ${r}` : ""}`), n && gn(), o)
      throw e;
    console.error(e);
  } else {
    if (s)
      throw e;
    console.error(e);
  }
}
const de = [];
let Be = -1;
const Dt = [];
let rt = null, xt = 0;
const ir = /* @__PURE__ */ Promise.resolve();
let Dn = null;
const zi = 100;
function Qi(e) {
  const t = Dn || ir;
  return e ? t.then(this ? e.bind(this) : e) : t;
}
function Xi(e) {
  let t = Be + 1, n = de.length;
  for (; t < n; ) {
    const o = t + n >>> 1, s = de[o], r = Qt(s);
    r < e || r === e && s.flags & 2 ? t = o + 1 : n = o;
  }
  return t;
}
function Ln(e) {
  if (!(e.flags & 1)) {
    const t = Qt(e), n = de[de.length - 1];
    !n || // fast path when the job id is larger than the tail
    !(e.flags & 2) && t >= Qt(n) ? de.push(e) : de.splice(Xi(t), 0, e), e.flags |= 1, lr();
  }
}
function lr() {
  Dn || (Dn = ir.then(fr));
}
function cr(e) {
  T(e) ? Dt.push(...e) : rt && e.id === -1 ? rt.splice(xt + 1, 0, e) : e.flags & 1 || (Dt.push(e), e.flags |= 1), lr();
}
function es(e, t, n = Be + 1) {
  for (process.env.NODE_ENV !== "production" && (t = t || /* @__PURE__ */ new Map()); n < de.length; n++) {
    const o = de[n];
    if (o && o.flags & 2) {
      if (e && o.id !== e.uid || process.env.NODE_ENV !== "production" && Po(t, o))
        continue;
      de.splice(n, 1), n--, o.flags & 4 && (o.flags &= -2), o(), o.flags & 4 || (o.flags &= -2);
    }
  }
}
function ur(e) {
  if (Dt.length) {
    const t = [...new Set(Dt)].sort(
      (n, o) => Qt(n) - Qt(o)
    );
    if (Dt.length = 0, rt) {
      rt.push(...t);
      return;
    }
    for (rt = t, process.env.NODE_ENV !== "production" && (e = e || /* @__PURE__ */ new Map()), xt = 0; xt < rt.length; xt++) {
      const n = rt[xt];
      process.env.NODE_ENV !== "production" && Po(e, n) || (n.flags & 4 && (n.flags &= -2), n.flags & 8 || n(), n.flags &= -2);
    }
    rt = null, xt = 0;
  }
}
const Qt = (e) => e.id == null ? e.flags & 2 ? -1 : 1 / 0 : e.id;
function fr(e) {
  process.env.NODE_ENV !== "production" && (e = e || /* @__PURE__ */ new Map());
  const t = process.env.NODE_ENV !== "production" ? (n) => Po(e, n) : ie;
  try {
    for (Be = 0; Be < de.length; Be++) {
      const n = de[Be];
      if (n && !(n.flags & 8)) {
        if (process.env.NODE_ENV !== "production" && t(n))
          continue;
        n.flags & 4 && (n.flags &= -2), Tt(
          n,
          n.i,
          n.i ? 15 : 14
        ), n.flags & 4 || (n.flags &= -2);
      }
    }
  } finally {
    for (; Be < de.length; Be++) {
      const n = de[Be];
      n && (n.flags &= -2);
    }
    Be = -1, de.length = 0, ur(e), Dn = null, (de.length || Dt.length) && fr(e);
  }
}
function Po(e, t) {
  const n = e.get(t) || 0;
  if (n > zi) {
    const o = t.i, s = o && Gr(o.type);
    return on(
      `Maximum recursive updates exceeded${s ? ` in component <${s}>` : ""}. This means you have a reactive effect that is mutating its own dependencies and thus recursively triggering itself. Possible sources include component template, render function, updated hook or watcher source function.`,
      null,
      10
    ), !0;
  }
  return e.set(t, n + 1), !1;
}
let Ge = !1;
const vn = /* @__PURE__ */ new Map();
process.env.NODE_ENV !== "production" && (nn().__VUE_HMR_RUNTIME__ = {
  createRecord: Xn(ar),
  rerender: Xn(tl),
  reload: Xn(nl)
});
const Et = /* @__PURE__ */ new Map();
function Zi(e) {
  const t = e.type.__hmrId;
  let n = Et.get(t);
  n || (ar(t, e.type), n = Et.get(t)), n.instances.add(e);
}
function el(e) {
  Et.get(e.type.__hmrId).instances.delete(e);
}
function ar(e, t) {
  return Et.has(e) ? !1 : (Et.set(e, {
    initialDef: Vn(t),
    instances: /* @__PURE__ */ new Set()
  }), !0);
}
function Vn(e) {
  return qr(e) ? e.__vccOpts : e;
}
function tl(e, t) {
  const n = Et.get(e);
  n && (n.initialDef.render = t, [...n.instances].forEach((o) => {
    t && (o.render = t, Vn(o.type).render = t), o.renderCache = [], Ge = !0, o.job.flags & 8 || o.update(), Ge = !1;
  }));
}
function nl(e, t) {
  const n = Et.get(e);
  if (!n) return;
  t = Vn(t), ts(n.initialDef, t);
  const o = [...n.instances];
  for (let s = 0; s < o.length; s++) {
    const r = o[s], l = Vn(r.type);
    let i = vn.get(l);
    i || (l !== n.initialDef && ts(l, t), vn.set(l, i = /* @__PURE__ */ new Set())), i.add(r), r.appContext.propsCache.delete(r.type), r.appContext.emitsCache.delete(r.type), r.appContext.optionsCache.delete(r.type), r.ceReload ? (i.add(r), r.ceReload(t.styles), i.delete(r)) : r.parent ? Ln(() => {
      r.job.flags & 8 || (Ge = !0, r.parent.update(), Ge = !1, i.delete(r));
    }) : r.appContext.reload ? r.appContext.reload() : typeof window < "u" ? window.location.reload() : console.warn(
      "[HMR] Root or manually mounted instance modified. Full reload required."
    ), r.root.ce && r !== r.root && r.root.ce._removeChildStyle(l);
  }
  cr(() => {
    vn.clear();
  });
}
function ts(e, t) {
  Z(e, t);
  for (const n in e)
    n !== "__file" && !(n in t) && delete e[n];
}
function Xn(e) {
  return (t, n) => {
    try {
      return e(t, n);
    } catch (o) {
      console.error(o), console.warn(
        "[HMR] Something went wrong during Vue component hot-reload. Full reload required."
      );
    }
  };
}
let Te, kt = [], fo = !1;
function sn(e, ...t) {
  Te ? Te.emit(e, ...t) : fo || kt.push({ event: e, args: t });
}
function Ao(e, t) {
  var n, o;
  Te = e, Te ? (Te.enabled = !0, kt.forEach(({ event: s, args: r }) => Te.emit(s, ...r)), kt = []) : /* handle late devtools injection - only do this if we are in an actual */ /* browser environment to avoid the timer handle stalling test runner exit */ /* (#4815) */ typeof window < "u" && // some envs mock window but not fully
  window.HTMLElement && // also exclude jsdom
  // eslint-disable-next-line no-restricted-syntax
  !((o = (n = window.navigator) == null ? void 0 : n.userAgent) != null && o.includes("jsdom")) ? ((t.__VUE_DEVTOOLS_HOOK_REPLAY__ = t.__VUE_DEVTOOLS_HOOK_REPLAY__ || []).push((r) => {
    Ao(r, t);
  }), setTimeout(() => {
    Te || (t.__VUE_DEVTOOLS_HOOK_REPLAY__ = null, fo = !0, kt = []);
  }, 3e3)) : (fo = !0, kt = []);
}
function ol(e, t) {
  sn("app:init", e, t, {
    Fragment: Ne,
    Text: rn,
    Comment: we,
    Static: En
  });
}
function sl(e) {
  sn("app:unmount", e);
}
const rl = /* @__PURE__ */ Mo(
  "component:added"
  /* COMPONENT_ADDED */
), pr = /* @__PURE__ */ Mo(
  "component:updated"
  /* COMPONENT_UPDATED */
), il = /* @__PURE__ */ Mo(
  "component:removed"
  /* COMPONENT_REMOVED */
), ll = (e) => {
  Te && typeof Te.cleanupBuffer == "function" && // remove the component if it wasn't buffered
  !Te.cleanupBuffer(e) && il(e);
};
// @__NO_SIDE_EFFECTS__
function Mo(e) {
  return (t) => {
    sn(
      e,
      t.appContext.app,
      t.uid,
      t.parent ? t.parent.uid : void 0,
      t
    );
  };
}
const cl = /* @__PURE__ */ dr(
  "perf:start"
  /* PERFORMANCE_START */
), ul = /* @__PURE__ */ dr(
  "perf:end"
  /* PERFORMANCE_END */
);
function dr(e) {
  return (t, n, o) => {
    sn(e, t.appContext.app, t.uid, t, n, o);
  };
}
function fl(e, t, n) {
  sn(
    "component:emit",
    e.appContext.app,
    e,
    t,
    n
  );
}
let be = null, hr = null;
function Sn(e) {
  const t = be;
  return be = e, hr = e && e.type.__scopeId || null, t;
}
function al(e, t = be, n) {
  if (!t || e._n)
    return e;
  const o = (...s) => {
    o._d && Pn(-1);
    const r = Sn(t);
    let l;
    try {
      l = e(...s);
    } finally {
      Sn(r), o._d && Pn(1);
    }
    return process.env.NODE_ENV !== "production" && pr(t), l;
  };
  return o._n = !0, o._c = !0, o._d = !0, o;
}
function gr(e) {
  Zr(e) && b("Do not use built-in directive ids as custom directive id: " + e);
}
function at(e, t, n, o) {
  const s = e.dirs, r = t && t.dirs;
  for (let l = 0; l < s.length; l++) {
    const i = s[l];
    r && (i.oldValue = r[l].value);
    let u = i.dir[o];
    u && (Ae(), Ye(u, n, 8, [
      e.el,
      i,
      e,
      t
    ]), Me());
  }
}
function pl(e, t) {
  if (process.env.NODE_ENV !== "production" && (!se || se.isMounted) && b("provide() can only be used inside setup()."), se) {
    let n = se.provides;
    const o = se.parent && se.parent.provides;
    o === n && (n = se.provides = Object.create(o)), n[e] = t;
  }
}
function mn(e, t, n = !1) {
  const o = Br();
  if (o || Vt) {
    let s = Vt ? Vt._context.provides : o ? o.parent == null || o.ce ? o.vnode.appContext && o.vnode.appContext.provides : o.parent.provides : void 0;
    if (s && e in s)
      return s[e];
    if (arguments.length > 1)
      return n && M(t) ? t.call(o && o.proxy) : t;
    process.env.NODE_ENV !== "production" && b(`injection "${String(e)}" not found.`);
  } else process.env.NODE_ENV !== "production" && b("inject() can only be used inside setup() or functional components.");
}
const dl = /* @__PURE__ */ Symbol.for("v-scx"), hl = () => {
  {
    const e = mn(dl);
    return e || process.env.NODE_ENV !== "production" && b(
      "Server rendering context not provided. Make sure to only call useSSRContext() conditionally in the server build."
    ), e;
  }
};
function Zn(e, t, n) {
  return process.env.NODE_ENV !== "production" && !M(t) && b(
    "`watch(fn, options?)` signature has been moved to a separate API. Use `watchEffect(fn, options?)` instead. `watch` now only supports `watch(source, cb, options?) signature."
  ), vr(e, t, n);
}
function vr(e, t, n = Y) {
  const { immediate: o, deep: s, flush: r, once: l } = n;
  process.env.NODE_ENV !== "production" && !t && (o !== void 0 && b(
    'watch() "immediate" option is only respected when using the watch(source, callback, options?) signature.'
  ), s !== void 0 && b(
    'watch() "deep" option is only respected when using the watch(source, callback, options?) signature.'
  ), l !== void 0 && b(
    'watch() "once" option is only respected when using the watch(source, callback, options?) signature.'
  ));
  const i = Z({}, n);
  process.env.NODE_ENV !== "production" && (i.onWarn = b);
  const u = t && o || !t && r !== "post";
  let d;
  if (Zt) {
    if (r === "sync") {
      const O = hl();
      d = O.__watcherHandles || (O.__watcherHandles = []);
    } else if (!u) {
      const O = () => {
      };
      return O.stop = ie, O.resume = ie, O.pause = ie, O;
    }
  }
  const a = se;
  i.call = (O, $, D) => Ye(O, a, $, D);
  let p = !1;
  r === "post" ? i.scheduler = (O) => {
    Ee(O, a && a.suspense);
  } : r !== "sync" && (p = !0, i.scheduler = (O, $) => {
    $ ? O() : Ln(O);
  }), i.augmentJob = (O) => {
    t && (O.flags |= 4), p && (O.flags |= 2, a && (O.id = a.uid, O.i = a));
  };
  const g = Ui(e, t, i);
  return Zt && (d ? d.push(g) : u && g()), g;
}
function gl(e, t, n) {
  const o = this.proxy, s = X(e) ? e.includes(".") ? mr(o, e) : () => o[e] : e.bind(o, o);
  let r;
  M(t) ? r = t : (r = t.handler, n = t);
  const l = ln(this), i = vr(s, r.bind(o), n);
  return l(), i;
}
function mr(e, t) {
  const n = t.split(".");
  return () => {
    let o = e;
    for (let s = 0; s < n.length && o; s++)
      o = o[n[s]];
    return o;
  };
}
const vl = /* @__PURE__ */ Symbol("_vte"), ml = (e) => e.__isTeleport, _l = /* @__PURE__ */ Symbol("_leaveCb");
function Ro(e, t) {
  e.shapeFlag & 6 && e.component ? (e.transition = t, Ro(e.component.subTree, t)) : e.shapeFlag & 128 ? (e.ssContent.transition = t.clone(e.ssContent), e.ssFallback.transition = t.clone(e.ssFallback)) : e.transition = t;
}
// @__NO_SIDE_EFFECTS__
function El(e, t) {
  return M(e) ? (
    // #8236: extend call and options.name access are considered side-effects
    // by Rollup, so we have to wrap it in a pure-annotated IIFE.
    Z({ name: e.name }, t, { setup: e })
  ) : e;
}
function _r(e) {
  e.ids = [e.ids[0] + e.ids[2]++ + "-", 0, 0];
}
const ns = /* @__PURE__ */ new WeakSet();
function os(e, t) {
  let n;
  return !!((n = Object.getOwnPropertyDescriptor(e, t)) && !n.configurable);
}
const Cn = /* @__PURE__ */ new WeakMap();
function Gt(e, t, n, o, s = !1) {
  if (T(e)) {
    e.forEach(
      (D, Q) => Gt(
        D,
        t && (T(t) ? t[Q] : t),
        n,
        o,
        s
      )
    );
    return;
  }
  if (qt(o) && !s) {
    o.shapeFlag & 512 && o.type.__asyncResolved && o.component.subTree.component && Gt(e, t, n, o.component.subTree);
    return;
  }
  const r = o.shapeFlag & 4 ? Uo(o.component) : o.el, l = s ? null : r, { i, r: u } = e;
  if (process.env.NODE_ENV !== "production" && !i) {
    b(
      "Missing ref owner context. ref cannot be used on hoisted vnodes. A vnode with ref must be created inside the render function."
    );
    return;
  }
  const d = t && t.r, a = i.refs === Y ? i.refs = {} : i.refs, p = i.setupState, g = /* @__PURE__ */ j(p), O = p === Y ? As : (D) => process.env.NODE_ENV !== "production" && (B(g, D) && !/* @__PURE__ */ ne(g[D]) && b(
    `Template ref "${D}" used on a non-ref value. It will not work in the production build.`
  ), ns.has(g[D])) || os(a, D) ? !1 : B(g, D), $ = (D, Q) => !(process.env.NODE_ENV !== "production" && ns.has(D) || Q && os(a, Q));
  if (d != null && d !== u) {
    if (ss(t), X(d))
      a[d] = null, O(d) && (p[d] = null);
    else if (/* @__PURE__ */ ne(d)) {
      const D = t;
      $(d, D.k) && (d.value = null), D.k && (a[D.k] = null);
    }
  }
  if (M(u))
    Tt(u, i, 12, [l, a]);
  else {
    const D = X(u), Q = /* @__PURE__ */ ne(u);
    if (D || Q) {
      const J = () => {
        if (e.f) {
          const K = D ? O(u) ? p[u] : a[u] : $(u) || !e.k ? u.value : a[e.k];
          if (s)
            T(K) && No(K, r);
          else if (T(K))
            K.includes(r) || K.push(r);
          else if (D)
            a[u] = [r], O(u) && (p[u] = a[u]);
          else {
            const L = [r];
            $(u, e.k) && (u.value = L), e.k && (a[e.k] = L);
          }
        } else D ? (a[u] = l, O(u) && (p[u] = l)) : Q ? ($(u, e.k) && (u.value = l), e.k && (a[e.k] = l)) : process.env.NODE_ENV !== "production" && b("Invalid template ref type:", u, `(${typeof u})`);
      };
      if (l) {
        const K = () => {
          J(), Cn.delete(e);
        };
        K.id = -1, Cn.set(e, K), Ee(K, n);
      } else
        ss(e), J();
    } else process.env.NODE_ENV !== "production" && b("Invalid template ref type:", u, `(${typeof u})`);
  }
}
function ss(e) {
  const t = Cn.get(e);
  t && (t.flags |= 8, Cn.delete(e));
}
nn().requestIdleCallback;
nn().cancelIdleCallback;
const qt = (e) => !!e.type.__asyncLoader, Fo = (e) => e.type.__isKeepAlive;
function yl(e, t) {
  Er(e, "a", t);
}
function Nl(e, t) {
  Er(e, "da", t);
}
function Er(e, t, n = se) {
  const o = e.__wdc || (e.__wdc = () => {
    let s = n;
    for (; s; ) {
      if (s.isDeactivated)
        return;
      s = s.parent;
    }
    return e();
  });
  if (Wn(t, o, n), n) {
    let s = n.parent;
    for (; s && s.parent; )
      Fo(s.parent.vnode) && bl(o, t, n, s), s = s.parent;
  }
}
function bl(e, t, n, o) {
  const s = Wn(
    t,
    e,
    o,
    !0
    /* prepend */
  );
  jo(() => {
    No(o[t], s);
  }, n);
}
function Wn(e, t, n = se, o = !1) {
  if (n) {
    const s = n[e] || (n[e] = []), r = t.__weh || (t.__weh = (...l) => {
      Ae();
      const i = ln(n), u = Ye(t, n, e, l);
      return i(), Me(), u;
    });
    return o ? s.unshift(r) : s.push(r), r;
  } else if (process.env.NODE_ENV !== "production") {
    const s = dt(Io[e].replace(/ hook$/, ""));
    b(
      `${s} is called when there is no active component instance to be associated with. Lifecycle injection APIs can only be used during execution of setup(). If you are using async setup(), make sure to register lifecycle hooks before the first await statement.`
    );
  }
}
const nt = (e) => (t, n = se) => {
  (!Zt || e === "sp") && Wn(e, (...o) => t(...o), n);
}, Ol = nt("bm"), yr = nt("m"), xl = nt(
  "bu"
), wl = nt("u"), Dl = nt(
  "bum"
), jo = nt("um"), Vl = nt(
  "sp"
), Sl = nt("rtg"), Cl = nt("rtc");
function Tl(e, t = se) {
  Wn("ec", e, t);
}
const $l = /* @__PURE__ */ Symbol.for("v-ndc");
function eo(e, t, n, o) {
  let s;
  const r = n, l = T(e);
  if (l || X(e)) {
    const i = l && /* @__PURE__ */ ct(e);
    let u = !1, d = !1;
    i && (u = !/* @__PURE__ */ he(e), d = /* @__PURE__ */ Re(e), e = jn(e)), s = new Array(e.length);
    for (let a = 0, p = e.length; a < p; a++)
      s[a] = t(
        u ? d ? St(Fe(e[a])) : Fe(e[a]) : e[a],
        a,
        void 0,
        r
      );
  } else if (typeof e == "number") {
    process.env.NODE_ENV !== "production" && !Number.isInteger(e) && b(`The v-for range expect an integer value but got ${e}.`), s = new Array(e);
    for (let i = 0; i < e; i++)
      s[i] = t(i + 1, i, void 0, r);
  } else if (U(e))
    if (e[Symbol.iterator])
      s = Array.from(
        e,
        (i, u) => t(i, u, void 0, r)
      );
    else {
      const i = Object.keys(e);
      s = new Array(i.length);
      for (let u = 0, d = i.length; u < d; u++) {
        const a = i[u];
        s[u] = t(e[a], a, u, r);
      }
    }
  else
    s = [];
  return s;
}
const ao = (e) => e ? Ur(e) ? Uo(e) : ao(e.parent) : null, _t = (
  // Move PURE marker to new line to workaround compiler discarding it
  // due to type annotation
  /* @__PURE__ */ Z(/* @__PURE__ */ Object.create(null), {
    $: (e) => e,
    $el: (e) => e.vnode.el,
    $data: (e) => e.data,
    $props: (e) => process.env.NODE_ENV !== "production" ? /* @__PURE__ */ Ke(e.props) : e.props,
    $attrs: (e) => process.env.NODE_ENV !== "production" ? /* @__PURE__ */ Ke(e.attrs) : e.attrs,
    $slots: (e) => process.env.NODE_ENV !== "production" ? /* @__PURE__ */ Ke(e.slots) : e.slots,
    $refs: (e) => process.env.NODE_ENV !== "production" ? /* @__PURE__ */ Ke(e.refs) : e.refs,
    $parent: (e) => ao(e.parent),
    $root: (e) => ao(e.root),
    $host: (e) => e.ce,
    $emit: (e) => e.emit,
    $options: (e) => Or(e),
    $forceUpdate: (e) => e.f || (e.f = () => {
      Ln(e.update);
    }),
    $nextTick: (e) => e.n || (e.n = Qi.bind(e.proxy)),
    $watch: (e) => gl.bind(e)
  })
), Ho = (e) => e === "_" || e === "$", to = (e, t) => e !== Y && !e.__isScriptSetup && B(e, t), Nr = {
  get({ _: e }, t) {
    if (t === "__v_skip")
      return !0;
    const { ctx: n, setupState: o, data: s, props: r, accessCache: l, type: i, appContext: u } = e;
    if (process.env.NODE_ENV !== "production" && t === "__isVue")
      return !0;
    if (t[0] !== "$") {
      const g = l[t];
      if (g !== void 0)
        switch (g) {
          case 1:
            return o[t];
          case 2:
            return s[t];
          case 4:
            return n[t];
          case 3:
            return r[t];
        }
      else {
        if (to(o, t))
          return l[t] = 1, o[t];
        if (s !== Y && B(s, t))
          return l[t] = 2, s[t];
        if (B(r, t))
          return l[t] = 3, r[t];
        if (n !== Y && B(n, t))
          return l[t] = 4, n[t];
        po && (l[t] = 0);
      }
    }
    const d = _t[t];
    let a, p;
    if (d)
      return t === "$attrs" ? (re(e.attrs, "get", ""), process.env.NODE_ENV !== "production" && $n()) : process.env.NODE_ENV !== "production" && t === "$slots" && re(e, "get", t), d(e);
    if (
      // css module (injected by vue-loader)
      (a = i.__cssModules) && (a = a[t])
    )
      return a;
    if (n !== Y && B(n, t))
      return l[t] = 4, n[t];
    if (
      // global properties
      p = u.config.globalProperties, B(p, t)
    )
      return p[t];
    process.env.NODE_ENV !== "production" && be && (!X(t) || // #1091 avoid internal isRef/isVNode checks on component instance leading
    // to infinite warning loop
    t.indexOf("__v") !== 0) && (s !== Y && Ho(t[0]) && B(s, t) ? b(
      `Property ${JSON.stringify(
        t
      )} must be accessed via $data because it starts with a reserved character ("$" or "_") and is not proxied on the render context.`
    ) : e === be && b(
      `Property ${JSON.stringify(t)} was accessed during render but is not defined on instance.`
    ));
  },
  set({ _: e }, t, n) {
    const { data: o, setupState: s, ctx: r } = e;
    return to(s, t) ? (s[t] = n, !0) : process.env.NODE_ENV !== "production" && s.__isScriptSetup && B(s, t) ? (b(`Cannot mutate <script setup> binding "${t}" from Options API.`), !1) : o !== Y && B(o, t) ? (o[t] = n, !0) : B(e.props, t) ? (process.env.NODE_ENV !== "production" && b(`Attempting to mutate prop "${t}". Props are readonly.`), !1) : t[0] === "$" && t.slice(1) in e ? (process.env.NODE_ENV !== "production" && b(
      `Attempting to mutate public property "${t}". Properties starting with $ are reserved and readonly.`
    ), !1) : (process.env.NODE_ENV !== "production" && t in e.appContext.config.globalProperties ? Object.defineProperty(r, t, {
      enumerable: !0,
      configurable: !0,
      value: n
    }) : r[t] = n, !0);
  },
  has({
    _: { data: e, setupState: t, accessCache: n, ctx: o, appContext: s, props: r, type: l }
  }, i) {
    let u;
    return !!(n[i] || e !== Y && i[0] !== "$" && B(e, i) || to(t, i) || B(r, i) || B(o, i) || B(_t, i) || B(s.config.globalProperties, i) || (u = l.__cssModules) && u[i]);
  },
  defineProperty(e, t, n) {
    return n.get != null ? e._.accessCache[t] = 0 : B(n, "value") && this.set(e, t, n.value, null), Reflect.defineProperty(e, t, n);
  }
};
process.env.NODE_ENV !== "production" && (Nr.ownKeys = (e) => (b(
  "Avoid app logic that relies on enumerating keys on a component instance. The keys will be empty in production mode to avoid performance overhead."
), Reflect.ownKeys(e)));
function Il(e) {
  const t = {};
  return Object.defineProperty(t, "_", {
    configurable: !0,
    enumerable: !1,
    get: () => e
  }), Object.keys(_t).forEach((n) => {
    Object.defineProperty(t, n, {
      configurable: !0,
      enumerable: !1,
      get: () => _t[n](e),
      // intercepted by the proxy so no need for implementation,
      // but needed to prevent set errors
      set: ie
    });
  }), t;
}
function Pl(e) {
  const {
    ctx: t,
    propsOptions: [n]
  } = e;
  n && Object.keys(n).forEach((o) => {
    Object.defineProperty(t, o, {
      enumerable: !0,
      configurable: !0,
      get: () => e.props[o],
      set: ie
    });
  });
}
function Al(e) {
  const { ctx: t, setupState: n } = e;
  Object.keys(/* @__PURE__ */ j(n)).forEach((o) => {
    if (!n.__isScriptSetup) {
      if (Ho(o[0])) {
        b(
          `setup() return property ${JSON.stringify(
            o
          )} should not start with "$" or "_" which are reserved prefixes for Vue internals.`
        );
        return;
      }
      Object.defineProperty(t, o, {
        enumerable: !0,
        configurable: !0,
        get: () => n[o],
        set: ie
      });
    }
  });
}
function rs(e) {
  return T(e) ? e.reduce(
    (t, n) => (t[n] = null, t),
    {}
  ) : e;
}
function Ml() {
  const e = /* @__PURE__ */ Object.create(null);
  return (t, n) => {
    e[n] ? b(`${t} property "${n}" is already defined in ${e[n]}.`) : e[n] = t;
  };
}
let po = !0;
function Rl(e) {
  const t = Or(e), n = e.proxy, o = e.ctx;
  po = !1, t.beforeCreate && is(t.beforeCreate, e, "bc");
  const {
    // state
    data: s,
    computed: r,
    methods: l,
    watch: i,
    provide: u,
    inject: d,
    // lifecycle
    created: a,
    beforeMount: p,
    mounted: g,
    beforeUpdate: O,
    updated: $,
    activated: D,
    deactivated: Q,
    beforeDestroy: J,
    beforeUnmount: K,
    destroyed: L,
    unmounted: ue,
    render: C,
    renderTracked: ee,
    renderTriggered: fe,
    errorCaptured: le,
    serverPrefetch: ce,
    // public API
    expose: Se,
    inheritAttrs: De,
    // assets
    components: V,
    directives: I,
    filters: P
  } = t, k = process.env.NODE_ENV !== "production" ? Ml() : null;
  if (process.env.NODE_ENV !== "production") {
    const [H] = e.propsOptions;
    if (H)
      for (const F in H)
        k("Props", F);
  }
  if (d && Fl(d, o, k), l)
    for (const H in l) {
      const F = l[H];
      M(F) ? (process.env.NODE_ENV !== "production" ? Object.defineProperty(o, H, {
        value: F.bind(n),
        configurable: !0,
        enumerable: !0,
        writable: !0
      }) : o[H] = F.bind(n), process.env.NODE_ENV !== "production" && k("Methods", H)) : process.env.NODE_ENV !== "production" && b(
        `Method "${H}" has type "${typeof F}" in the component definition. Did you reference the function correctly?`
      );
    }
  if (s) {
    process.env.NODE_ENV !== "production" && !M(s) && b(
      "The data option must be a function. Plain object usage is no longer supported."
    );
    const H = s.call(n, n);
    if (process.env.NODE_ENV !== "production" && bo(H) && b(
      "data() returned a Promise - note data() cannot be async; If you intend to perform data fetching before component renders, use async setup() + <Suspense>."
    ), !U(H))
      process.env.NODE_ENV !== "production" && b("data() should return an object.");
    else if (e.data = /* @__PURE__ */ $o(H), process.env.NODE_ENV !== "production")
      for (const F in H)
        k("Data", F), Ho(F[0]) || Object.defineProperty(o, F, {
          configurable: !0,
          enumerable: !0,
          get: () => H[F],
          set: ie
        });
  }
  if (po = !0, r)
    for (const H in r) {
      const F = r[H], ge = M(F) ? F.bind(n, n) : M(F.get) ? F.get.bind(n, n) : ie;
      process.env.NODE_ENV !== "production" && ge === ie && b(`Computed property "${H}" has no getter.`);
      const Un = !M(F) && M(F.set) ? F.set.bind(n) : process.env.NODE_ENV !== "production" ? () => {
        b(
          `Write operation failed: computed property "${H}" is readonly.`
        );
      } : ie, $t = Wt({
        get: ge,
        set: Un
      });
      Object.defineProperty(o, H, {
        enumerable: !0,
        configurable: !0,
        get: () => $t.value,
        set: (yt) => $t.value = yt
      }), process.env.NODE_ENV !== "production" && k("Computed", H);
    }
  if (i)
    for (const H in i)
      br(i[H], o, n, H);
  if (u) {
    const H = M(u) ? u.call(n) : u;
    Reflect.ownKeys(H).forEach((F) => {
      pl(F, H[F]);
    });
  }
  a && is(a, e, "c");
  function oe(H, F) {
    T(F) ? F.forEach((ge) => H(ge.bind(n))) : F && H(F.bind(n));
  }
  if (oe(Ol, p), oe(yr, g), oe(xl, O), oe(wl, $), oe(yl, D), oe(Nl, Q), oe(Tl, le), oe(Cl, ee), oe(Sl, fe), oe(Dl, K), oe(jo, ue), oe(Vl, ce), T(Se))
    if (Se.length) {
      const H = e.exposed || (e.exposed = {});
      Se.forEach((F) => {
        Object.defineProperty(H, F, {
          get: () => n[F],
          set: (ge) => n[F] = ge,
          enumerable: !0
        });
      });
    } else e.exposed || (e.exposed = {});
  C && e.render === ie && (e.render = C), De != null && (e.inheritAttrs = De), V && (e.components = V), I && (e.directives = I), ce && _r(e);
}
function Fl(e, t, n = ie) {
  T(e) && (e = ho(e));
  for (const o in e) {
    const s = e[o];
    let r;
    U(s) ? "default" in s ? r = mn(
      s.from || o,
      s.default,
      !0
    ) : r = mn(s.from || o) : r = mn(s), /* @__PURE__ */ ne(r) ? Object.defineProperty(t, o, {
      enumerable: !0,
      configurable: !0,
      get: () => r.value,
      set: (l) => r.value = l
    }) : t[o] = r, process.env.NODE_ENV !== "production" && n("Inject", o);
  }
}
function is(e, t, n) {
  Ye(
    T(e) ? e.map((o) => o.bind(t.proxy)) : e.bind(t.proxy),
    t,
    n
  );
}
function br(e, t, n, o) {
  let s = o.includes(".") ? mr(n, o) : () => n[o];
  if (X(e)) {
    const r = t[e];
    M(r) ? Zn(s, r) : process.env.NODE_ENV !== "production" && b(`Invalid watch handler specified by key "${e}"`, r);
  } else if (M(e))
    Zn(s, e.bind(n));
  else if (U(e))
    if (T(e))
      e.forEach((r) => br(r, t, n, o));
    else {
      const r = M(e.handler) ? e.handler.bind(n) : t[e.handler];
      M(r) ? Zn(s, r, e) : process.env.NODE_ENV !== "production" && b(`Invalid watch handler specified by key "${e.handler}"`, r);
    }
  else process.env.NODE_ENV !== "production" && b(`Invalid watch option: "${o}"`, e);
}
function Or(e) {
  const t = e.type, { mixins: n, extends: o } = t, {
    mixins: s,
    optionsCache: r,
    config: { optionMergeStrategies: l }
  } = e.appContext, i = r.get(t);
  let u;
  return i ? u = i : !s.length && !n && !o ? u = t : (u = {}, s.length && s.forEach(
    (d) => Tn(u, d, l, !0)
  ), Tn(u, t, l)), U(t) && r.set(t, u), u;
}
function Tn(e, t, n, o = !1) {
  const { mixins: s, extends: r } = t;
  r && Tn(e, r, n, !0), s && s.forEach(
    (l) => Tn(e, l, n, !0)
  );
  for (const l in t)
    if (o && l === "expose")
      process.env.NODE_ENV !== "production" && b(
        '"expose" option is ignored when declared in mixins or extends. It should only be declared in the base component itself.'
      );
    else {
      const i = jl[l] || n && n[l];
      e[l] = i ? i(e[l], t[l]) : t[l];
    }
  return e;
}
const jl = {
  data: ls,
  props: cs,
  emits: cs,
  // objects
  methods: Lt,
  computed: Lt,
  // lifecycle
  beforeCreate: pe,
  created: pe,
  beforeMount: pe,
  mounted: pe,
  beforeUpdate: pe,
  updated: pe,
  beforeDestroy: pe,
  beforeUnmount: pe,
  destroyed: pe,
  unmounted: pe,
  activated: pe,
  deactivated: pe,
  errorCaptured: pe,
  serverPrefetch: pe,
  // assets
  components: Lt,
  directives: Lt,
  // watch
  watch: kl,
  // provide / inject
  provide: ls,
  inject: Hl
};
function ls(e, t) {
  return t ? e ? function() {
    return Z(
      M(e) ? e.call(this, this) : e,
      M(t) ? t.call(this, this) : t
    );
  } : t : e;
}
function Hl(e, t) {
  return Lt(ho(e), ho(t));
}
function ho(e) {
  if (T(e)) {
    const t = {};
    for (let n = 0; n < e.length; n++)
      t[e[n]] = e[n];
    return t;
  }
  return e;
}
function pe(e, t) {
  return e ? [...new Set([].concat(e, t))] : t;
}
function Lt(e, t) {
  return e ? Z(/* @__PURE__ */ Object.create(null), e, t) : t;
}
function cs(e, t) {
  return e ? T(e) && T(t) ? [.../* @__PURE__ */ new Set([...e, ...t])] : Z(
    /* @__PURE__ */ Object.create(null),
    rs(e),
    rs(t ?? {})
  ) : t;
}
function kl(e, t) {
  if (!e) return t;
  if (!t) return e;
  const n = Z(/* @__PURE__ */ Object.create(null), e);
  for (const o in t)
    n[o] = pe(e[o], t[o]);
  return n;
}
function xr() {
  return {
    app: null,
    config: {
      isNativeTag: As,
      performance: !1,
      globalProperties: {},
      optionMergeStrategies: {},
      errorHandler: void 0,
      warnHandler: void 0,
      compilerOptions: {}
    },
    mixins: [],
    components: {},
    directives: {},
    provides: /* @__PURE__ */ Object.create(null),
    optionsCache: /* @__PURE__ */ new WeakMap(),
    propsCache: /* @__PURE__ */ new WeakMap(),
    emitsCache: /* @__PURE__ */ new WeakMap()
  };
}
let Ll = 0;
function Wl(e, t) {
  return function(o, s = null) {
    M(o) || (o = Z({}, o)), s != null && !U(s) && (process.env.NODE_ENV !== "production" && b("root props passed to app.mount() must be an object."), s = null);
    const r = xr(), l = /* @__PURE__ */ new WeakSet(), i = [];
    let u = !1;
    const d = r.app = {
      _uid: Ll++,
      _component: o,
      _props: s,
      _container: null,
      _context: r,
      _instance: null,
      version: ys,
      get config() {
        return r.config;
      },
      set config(a) {
        process.env.NODE_ENV !== "production" && b(
          "app.config cannot be replaced. Modify individual options instead."
        );
      },
      use(a, ...p) {
        return l.has(a) ? process.env.NODE_ENV !== "production" && b("Plugin has already been applied to target app.") : a && M(a.install) ? (l.add(a), a.install(d, ...p)) : M(a) ? (l.add(a), a(d, ...p)) : process.env.NODE_ENV !== "production" && b(
          'A plugin must either be a function or an object with an "install" function.'
        ), d;
      },
      mixin(a) {
        return r.mixins.includes(a) ? process.env.NODE_ENV !== "production" && b(
          "Mixin has already been applied to target app" + (a.name ? `: ${a.name}` : "")
        ) : r.mixins.push(a), d;
      },
      component(a, p) {
        return process.env.NODE_ENV !== "production" && Eo(a, r.config), p ? (process.env.NODE_ENV !== "production" && r.components[a] && b(`Component "${a}" has already been registered in target app.`), r.components[a] = p, d) : r.components[a];
      },
      directive(a, p) {
        return process.env.NODE_ENV !== "production" && gr(a), p ? (process.env.NODE_ENV !== "production" && r.directives[a] && b(`Directive "${a}" has already been registered in target app.`), r.directives[a] = p, d) : r.directives[a];
      },
      mount(a, p, g) {
        if (u)
          process.env.NODE_ENV !== "production" && b(
            "App has already been mounted.\nIf you want to remount the same app, move your app creation logic into a factory function and create fresh app instances for each mount - e.g. `const createMyApp = () => createApp(App)`"
          );
        else {
          process.env.NODE_ENV !== "production" && a.__vue_app__ && b(
            "There is already an app instance mounted on the host container.\n If you want to mount another app on the same host container, you need to unmount the previous app by calling `app.unmount()` first."
          );
          const O = d._ceVNode || xe(o, s);
          return O.appContext = r, g === !0 ? g = "svg" : g === !1 && (g = void 0), process.env.NODE_ENV !== "production" && (r.reload = () => {
            const $ = ft(O);
            $.el = null, e($, a, g);
          }), e(O, a, g), u = !0, d._container = a, a.__vue_app__ = d, process.env.NODE_ENV !== "production" && (d._instance = O.component, ol(d, ys)), Uo(O.component);
        }
      },
      onUnmount(a) {
        process.env.NODE_ENV !== "production" && typeof a != "function" && b(
          `Expected function as first argument to app.onUnmount(), but got ${typeof a}`
        ), i.push(a);
      },
      unmount() {
        u ? (Ye(
          i,
          d._instance,
          16
        ), e(null, d._container), process.env.NODE_ENV !== "production" && (d._instance = null, sl(d)), delete d._container.__vue_app__) : process.env.NODE_ENV !== "production" && b("Cannot unmount an app that is not mounted.");
      },
      provide(a, p) {
        return process.env.NODE_ENV !== "production" && a in r.provides && (B(r.provides, a) ? b(
          `App already provides property with key "${String(a)}". It will be overwritten with the new value.`
        ) : b(
          `App already provides property with key "${String(a)}" inherited from its parent element. It will be overwritten with the new value.`
        )), r.provides[a] = p, d;
      },
      runWithContext(a) {
        const p = Vt;
        Vt = d;
        try {
          return a();
        } finally {
          Vt = p;
        }
      }
    };
    return d;
  };
}
let Vt = null;
const Bl = (e, t) => t === "modelValue" || t === "model-value" ? e.modelModifiers : e[`${t}Modifiers`] || e[`${$e(t)}Modifiers`] || e[`${ut(t)}Modifiers`];
function Ul(e, t, ...n) {
  if (e.isUnmounted) return;
  const o = e.vnode.props || Y;
  if (process.env.NODE_ENV !== "production") {
    const {
      emitsOptions: a,
      propsOptions: [p]
    } = e;
    if (a)
      if (!(t in a))
        (!p || !(dt($e(t)) in p)) && b(
          `Component emitted event "${t}" but it is neither declared in the emits option nor as an "${dt($e(t))}" prop.`
        );
      else {
        const g = a[t];
        M(g) && (g(...n) || b(
          `Invalid event arguments: event validation failed for event "${t}".`
        ));
      }
  }
  let s = n;
  const r = t.startsWith("update:"), l = r && Bl(o, t.slice(7));
  if (l && (l.trim && (s = n.map((a) => X(a) ? a.trim() : a)), l.number && (s = n.map(ni))), process.env.NODE_ENV !== "production" && fl(e, t, s), process.env.NODE_ENV !== "production") {
    const a = t.toLowerCase();
    a !== t && o[dt(a)] && b(
      `Event "${a}" is emitted in component ${cn(
        e,
        e.type
      )} but the handler is registered for "${t}". Note that HTML attributes are case-insensitive and you cannot use v-on to listen to camelCase events when using in-DOM templates. You should probably use "${ut(
        t
      )}" instead of "${t}".`
    );
  }
  let i, u = o[i = dt(t)] || // also try camelCase event handler (#2249)
  o[i = dt($e(t))];
  !u && r && (u = o[i = dt(ut(t))]), u && Ye(
    u,
    e,
    6,
    s
  );
  const d = o[i + "Once"];
  if (d) {
    if (!e.emitted)
      e.emitted = {};
    else if (e.emitted[i])
      return;
    e.emitted[i] = !0, Ye(
      d,
      e,
      6,
      s
    );
  }
}
const Kl = /* @__PURE__ */ new WeakMap();
function wr(e, t, n = !1) {
  const o = n ? Kl : t.emitsCache, s = o.get(e);
  if (s !== void 0)
    return s;
  const r = e.emits;
  let l = {}, i = !1;
  if (!M(e)) {
    const u = (d) => {
      const a = wr(d, t, !0);
      a && (i = !0, Z(l, a));
    };
    !n && t.mixins.length && t.mixins.forEach(u), e.extends && u(e.extends), e.mixins && e.mixins.forEach(u);
  }
  return !r && !i ? (U(e) && o.set(e, null), null) : (T(r) ? r.forEach((u) => l[u] = null) : Z(l, r), U(e) && o.set(e, l), l);
}
function Bn(e, t) {
  return !e || !en(t) ? !1 : (t = t.slice(2).replace(/Once$/, ""), B(e, t[0].toLowerCase() + t.slice(1)) || B(e, ut(t)) || B(e, t));
}
let go = !1;
function $n() {
  go = !0;
}
function us(e) {
  const {
    type: t,
    vnode: n,
    proxy: o,
    withProxy: s,
    propsOptions: [r],
    slots: l,
    attrs: i,
    emit: u,
    render: d,
    renderCache: a,
    props: p,
    data: g,
    setupState: O,
    ctx: $,
    inheritAttrs: D
  } = e, Q = Sn(e);
  let J, K;
  process.env.NODE_ENV !== "production" && (go = !1);
  try {
    if (n.shapeFlag & 4) {
      const C = s || o, ee = process.env.NODE_ENV !== "production" && O.__isScriptSetup ? new Proxy(C, {
        get(fe, le, ce) {
          return b(
            `Property '${String(
              le
            )}' was accessed via 'this'. Avoid using 'this' in templates.`
          ), Reflect.get(fe, le, ce);
        }
      }) : C;
      J = Ce(
        d.call(
          ee,
          C,
          a,
          process.env.NODE_ENV !== "production" ? /* @__PURE__ */ Ke(p) : p,
          O,
          g,
          $
        )
      ), K = i;
    } else {
      const C = t;
      process.env.NODE_ENV !== "production" && i === p && $n(), J = Ce(
        C.length > 1 ? C(
          process.env.NODE_ENV !== "production" ? /* @__PURE__ */ Ke(p) : p,
          process.env.NODE_ENV !== "production" ? {
            get attrs() {
              return $n(), /* @__PURE__ */ Ke(i);
            },
            slots: l,
            emit: u
          } : { attrs: i, slots: l, emit: u }
        ) : C(
          process.env.NODE_ENV !== "production" ? /* @__PURE__ */ Ke(p) : p,
          null
        )
      ), K = t.props ? i : Gl(i);
    }
  } catch (C) {
    Yt.length = 0, on(C, e, 1), J = xe(we);
  }
  let L = J, ue;
  if (process.env.NODE_ENV !== "production" && J.patchFlag > 0 && J.patchFlag & 2048 && ([L, ue] = Dr(J)), K && D !== !1) {
    const C = Object.keys(K), { shapeFlag: ee } = L;
    if (C.length) {
      if (ee & 7)
        r && C.some(bn) && (K = ql(
          K,
          r
        )), L = ft(L, K, !1, !0);
      else if (process.env.NODE_ENV !== "production" && !go && L.type !== we) {
        const fe = Object.keys(i), le = [], ce = [];
        for (let Se = 0, De = fe.length; Se < De; Se++) {
          const V = fe[Se];
          en(V) ? bn(V) || le.push(V[2].toLowerCase() + V.slice(3)) : ce.push(V);
        }
        ce.length && b(
          `Extraneous non-props attributes (${ce.join(", ")}) were passed to component but could not be automatically inherited because component renders fragment or text or teleport root nodes.`
        ), le.length && b(
          `Extraneous non-emits event listeners (${le.join(", ")}) were passed to component but could not be automatically inherited because component renders fragment or text root nodes. If the listener is intended to be a component custom event listener only, declare it using the "emits" option.`
        );
      }
    }
  }
  return n.dirs && (process.env.NODE_ENV !== "production" && !fs(L) && b(
    "Runtime directive used on component with non-element root node. The directives will not function as intended."
  ), L = ft(L, null, !1, !0), L.dirs = L.dirs ? L.dirs.concat(n.dirs) : n.dirs), n.transition && (process.env.NODE_ENV !== "production" && !fs(L) && b(
    "Component inside <Transition> renders non-element root node that cannot be animated."
  ), Ro(L, n.transition)), process.env.NODE_ENV !== "production" && ue ? ue(L) : J = L, Sn(Q), J;
}
const Dr = (e) => {
  const t = e.children, n = e.dynamicChildren, o = ko(t, !1);
  if (o) {
    if (process.env.NODE_ENV !== "production" && o.patchFlag > 0 && o.patchFlag & 2048)
      return Dr(o);
  } else return [e, void 0];
  const s = t.indexOf(o), r = n ? n.indexOf(o) : -1, l = (i) => {
    t[s] = i, n && (r > -1 ? n[r] = i : i.patchFlag > 0 && (e.dynamicChildren = [...n, i]));
  };
  return [Ce(o), l];
};
function ko(e, t = !0) {
  let n;
  for (let o = 0; o < e.length; o++) {
    const s = e[o];
    if (Ct(s)) {
      if (s.type !== we || s.children === "v-if") {
        if (n)
          return;
        if (n = s, process.env.NODE_ENV !== "production" && t && n.patchFlag > 0 && n.patchFlag & 2048)
          return ko(n.children);
      }
    } else
      return;
  }
  return n;
}
const Gl = (e) => {
  let t;
  for (const n in e)
    (n === "class" || n === "style" || en(n)) && ((t || (t = {}))[n] = e[n]);
  return t;
}, ql = (e, t) => {
  const n = {};
  for (const o in e)
    (!bn(o) || !(o.slice(9) in t)) && (n[o] = e[o]);
  return n;
}, fs = (e) => e.shapeFlag & 7 || e.type === we;
function Yl(e, t, n) {
  const { props: o, children: s, component: r } = e, { props: l, children: i, patchFlag: u } = t, d = r.emitsOptions;
  if (process.env.NODE_ENV !== "production" && (s || i) && Ge || t.dirs || t.transition)
    return !0;
  if (n && u >= 0) {
    if (u & 1024)
      return !0;
    if (u & 16)
      return o ? as(o, l, d) : !!l;
    if (u & 8) {
      const a = t.dynamicProps;
      for (let p = 0; p < a.length; p++) {
        const g = a[p];
        if (Vr(l, o, g) && !Bn(d, g))
          return !0;
      }
    }
  } else
    return (s || i) && (!i || !i.$stable) ? !0 : o === l ? !1 : o ? l ? as(o, l, d) : !0 : !!l;
  return !1;
}
function as(e, t, n) {
  const o = Object.keys(t);
  if (o.length !== Object.keys(e).length)
    return !0;
  for (let s = 0; s < o.length; s++) {
    const r = o[s];
    if (Vr(t, e, r) && !Bn(n, r))
      return !0;
  }
  return !1;
}
function Vr(e, t, n) {
  const o = e[n], s = t[n];
  return n === "style" && U(o) && U(s) ? !Do(o, s) : o !== s;
}
function Jl({ vnode: e, parent: t }, n) {
  for (; t; ) {
    const o = t.subTree;
    if (o.suspense && o.suspense.activeBranch === e && (o.el = e.el), o === e)
      (e = t.vnode).el = n, t = t.parent;
    else
      break;
  }
}
const Sr = {}, Cr = () => Object.create(Sr), Tr = (e) => Object.getPrototypeOf(e) === Sr;
function zl(e, t, n, o = !1) {
  const s = {}, r = Cr();
  e.propsDefaults = /* @__PURE__ */ Object.create(null), $r(e, t, s, r);
  for (const l in e.propsOptions[0])
    l in s || (s[l] = void 0);
  process.env.NODE_ENV !== "production" && Pr(t || {}, s, e), n ? e.props = o ? s : /* @__PURE__ */ Ri(s) : e.type.props ? e.props = s : e.props = r, e.attrs = r;
}
function Ql(e) {
  for (; e; ) {
    if (e.type.__hmrId) return !0;
    e = e.parent;
  }
}
function Xl(e, t, n, o) {
  const {
    props: s,
    attrs: r,
    vnode: { patchFlag: l }
  } = e, i = /* @__PURE__ */ j(s), [u] = e.propsOptions;
  let d = !1;
  if (
    // always force full diff in dev
    // - #1942 if hmr is enabled with sfc component
    // - vite#872 non-sfc component used by sfc component
    !(process.env.NODE_ENV !== "production" && Ql(e)) && (o || l > 0) && !(l & 16)
  ) {
    if (l & 8) {
      const a = e.vnode.dynamicProps;
      for (let p = 0; p < a.length; p++) {
        let g = a[p];
        if (Bn(e.emitsOptions, g))
          continue;
        const O = t[g];
        if (u)
          if (B(r, g))
            O !== r[g] && (r[g] = O, d = !0);
          else {
            const $ = $e(g);
            s[$] = vo(
              u,
              i,
              $,
              O,
              e,
              !1
            );
          }
        else
          O !== r[g] && (r[g] = O, d = !0);
      }
    }
  } else {
    $r(e, t, s, r) && (d = !0);
    let a;
    for (const p in i)
      (!t || // for camelCase
      !B(t, p) && // it's possible the original props was passed in as kebab-case
      // and converted to camelCase (#955)
      ((a = ut(p)) === p || !B(t, a))) && (u ? n && // for camelCase
      (n[p] !== void 0 || // for kebab-case
      n[a] !== void 0) && (s[p] = vo(
        u,
        i,
        p,
        void 0,
        e,
        !0
      )) : delete s[p]);
    if (r !== i)
      for (const p in r)
        (!t || !B(t, p)) && (delete r[p], d = !0);
  }
  d && Ue(e.attrs, "set", ""), process.env.NODE_ENV !== "production" && Pr(t || {}, s, e);
}
function $r(e, t, n, o) {
  const [s, r] = e.propsOptions;
  let l = !1, i;
  if (t)
    for (let u in t) {
      if (Bt(u))
        continue;
      const d = t[u];
      let a;
      s && B(s, a = $e(u)) ? !r || !r.includes(a) ? n[a] = d : (i || (i = {}))[a] = d : Bn(e.emitsOptions, u) || (!(u in o) || d !== o[u]) && (o[u] = d, l = !0);
    }
  if (r) {
    const u = /* @__PURE__ */ j(n), d = i || Y;
    for (let a = 0; a < r.length; a++) {
      const p = r[a];
      n[p] = vo(
        s,
        u,
        p,
        d[p],
        e,
        !B(d, p)
      );
    }
  }
  return l;
}
function vo(e, t, n, o, s, r) {
  const l = e[n];
  if (l != null) {
    const i = B(l, "default");
    if (i && o === void 0) {
      const u = l.default;
      if (l.type !== Function && !l.skipFactory && M(u)) {
        const { propsDefaults: d } = s;
        if (n in d)
          o = d[n];
        else {
          const a = ln(s);
          o = d[n] = u.call(
            null,
            t
          ), a();
        }
      } else
        o = u;
      s.ce && s.ce._setProp(n, o);
    }
    l[
      0
      /* shouldCast */
    ] && (r && !i ? o = !1 : l[
      1
      /* shouldCastTrue */
    ] && (o === "" || o === ut(n)) && (o = !0));
  }
  return o;
}
const Zl = /* @__PURE__ */ new WeakMap();
function Ir(e, t, n = !1) {
  const o = n ? Zl : t.propsCache, s = o.get(e);
  if (s)
    return s;
  const r = e.props, l = {}, i = [];
  let u = !1;
  if (!M(e)) {
    const a = (p) => {
      u = !0;
      const [g, O] = Ir(p, t, !0);
      Z(l, g), O && i.push(...O);
    };
    !n && t.mixins.length && t.mixins.forEach(a), e.extends && a(e.extends), e.mixins && e.mixins.forEach(a);
  }
  if (!r && !u)
    return U(e) && o.set(e, wt), wt;
  if (T(r))
    for (let a = 0; a < r.length; a++) {
      process.env.NODE_ENV !== "production" && !X(r[a]) && b("props must be strings when using array syntax.", r[a]);
      const p = $e(r[a]);
      ps(p) && (l[p] = Y);
    }
  else if (r) {
    process.env.NODE_ENV !== "production" && !U(r) && b("invalid props options", r);
    for (const a in r) {
      const p = $e(a);
      if (ps(p)) {
        const g = r[a], O = l[p] = T(g) || M(g) ? { type: g } : Z({}, g), $ = O.type;
        let D = !1, Q = !0;
        if (T($))
          for (let J = 0; J < $.length; ++J) {
            const K = $[J], L = M(K) && K.name;
            if (L === "Boolean") {
              D = !0;
              break;
            } else L === "String" && (Q = !1);
          }
        else
          D = M($) && $.name === "Boolean";
        O[
          0
          /* shouldCast */
        ] = D, O[
          1
          /* shouldCastTrue */
        ] = Q, (D || B(O, "default")) && i.push(p);
      }
    }
  }
  const d = [l, i];
  return U(e) && o.set(e, d), d;
}
function ps(e) {
  return e[0] !== "$" && !Bt(e) ? !0 : (process.env.NODE_ENV !== "production" && b(`Invalid prop name: "${e}" is a reserved property.`), !1);
}
function ec(e) {
  return e === null ? "null" : typeof e == "function" ? e.name || "" : typeof e == "object" && e.constructor && e.constructor.name || "";
}
function Pr(e, t, n) {
  const o = /* @__PURE__ */ j(t), s = n.propsOptions[0], r = Object.keys(e).map((l) => $e(l));
  for (const l in s) {
    let i = s[l];
    i != null && tc(
      l,
      o[l],
      i,
      process.env.NODE_ENV !== "production" ? /* @__PURE__ */ Ke(o) : o,
      !r.includes(l)
    );
  }
}
function tc(e, t, n, o, s) {
  const { type: r, required: l, validator: i, skipCheck: u } = n;
  if (l && s) {
    b('Missing required prop: "' + e + '"');
    return;
  }
  if (!(t == null && !l)) {
    if (r != null && r !== !0 && !u) {
      let d = !1;
      const a = T(r) ? r : [r], p = [];
      for (let g = 0; g < a.length && !d; g++) {
        const { valid: O, expectedType: $ } = oc(t, a[g]);
        p.push($ || ""), d = O;
      }
      if (!d) {
        b(sc(e, t, p));
        return;
      }
    }
    i && !i(t, o) && b('Invalid prop: custom validator check failed for prop "' + e + '".');
  }
}
const nc = /* @__PURE__ */ tt(
  "String,Number,Boolean,Function,Symbol,BigInt"
);
function oc(e, t) {
  let n;
  const o = ec(t);
  if (o === "null")
    n = e === null;
  else if (nc(o)) {
    const s = typeof e;
    n = s === o.toLowerCase(), !n && s === "object" && (n = e instanceof t);
  } else o === "Object" ? n = U(e) : o === "Array" ? n = T(e) : n = e instanceof t;
  return {
    valid: n,
    expectedType: o
  };
}
function sc(e, t, n) {
  if (n.length === 0)
    return `Prop type [] for prop "${e}" won't match anything. Did you mean to use type Array instead?`;
  let o = `Invalid prop: type check failed for prop "${e}". Expected ${n.map(Rn).join(" | ")}`;
  const s = n[0], r = Oo(t), l = ds(t, s), i = ds(t, r);
  return n.length === 1 && hs(s) && !rc(s, r) && (o += ` with value ${l}`), o += `, got ${r} `, hs(r) && (o += `with value ${i}.`), o;
}
function ds(e, t) {
  return t === "String" ? `"${e}"` : t === "Number" ? `${Number(e)}` : `${e}`;
}
function hs(e) {
  return ["string", "number", "boolean"].some((n) => e.toLowerCase() === n);
}
function rc(...e) {
  return e.some((t) => t.toLowerCase() === "boolean");
}
const Lo = (e) => e === "_" || e === "_ctx" || e === "$stable", Wo = (e) => T(e) ? e.map(Ce) : [Ce(e)], ic = (e, t, n) => {
  if (t._n)
    return t;
  const o = al((...s) => (process.env.NODE_ENV !== "production" && se && !(n === null && be) && !(n && n.root !== se.root) && b(
    `Slot "${e}" invoked outside of the render function: this will not track dependencies used in the slot. Invoke the slot function inside the render function instead.`
  ), Wo(t(...s))), n);
  return o._c = !1, o;
}, Ar = (e, t, n) => {
  const o = e._ctx;
  for (const s in e) {
    if (Lo(s)) continue;
    const r = e[s];
    if (M(r))
      t[s] = ic(s, r, o);
    else if (r != null) {
      process.env.NODE_ENV !== "production" && b(
        `Non-function value encountered for slot "${s}". Prefer function slots for better performance.`
      );
      const l = Wo(r);
      t[s] = () => l;
    }
  }
}, Mr = (e, t) => {
  process.env.NODE_ENV !== "production" && !Fo(e.vnode) && b(
    "Non-function value encountered for default slot. Prefer function slots for better performance."
  );
  const n = Wo(t);
  e.slots.default = () => n;
}, mo = (e, t, n) => {
  for (const o in t)
    (n || !Lo(o)) && (e[o] = t[o]);
}, lc = (e, t, n) => {
  const o = e.slots = Cr();
  if (e.vnode.shapeFlag & 32) {
    const s = t._;
    s ? (mo(o, t, n), n && On(o, "_", s, !0)) : Ar(t, o);
  } else t && Mr(e, t);
}, cc = (e, t, n) => {
  const { vnode: o, slots: s } = e;
  let r = !0, l = Y;
  if (o.shapeFlag & 32) {
    const i = t._;
    i ? process.env.NODE_ENV !== "production" && Ge ? (mo(s, t, n), Ue(e, "set", "$slots")) : n && i === 1 ? r = !1 : mo(s, t, n) : (r = !t.$stable, Ar(t, s)), l = t;
  } else t && (Mr(e, t), l = { default: 1 });
  if (r)
    for (const i in s)
      !Lo(i) && l[i] == null && delete s[i];
};
let jt, Xe;
function bt(e, t) {
  e.appContext.config.performance && In() && Xe.mark(`vue-${t}-${e.uid}`), process.env.NODE_ENV !== "production" && cl(e, t, In() ? Xe.now() : Date.now());
}
function Ot(e, t) {
  if (e.appContext.config.performance && In()) {
    const n = `vue-${t}-${e.uid}`, o = n + ":end", s = `<${cn(e, e.type)}> ${t}`;
    Xe.mark(o), Xe.measure(s, n, o), Xe.clearMeasures(s), Xe.clearMarks(n), Xe.clearMarks(o);
  }
  process.env.NODE_ENV !== "production" && ul(e, t, In() ? Xe.now() : Date.now());
}
function In() {
  return jt !== void 0 || (typeof window < "u" && window.performance ? (jt = !0, Xe = window.performance) : jt = !1), jt;
}
function uc() {
  const e = [];
  if (process.env.NODE_ENV !== "production" && e.length) {
    const t = e.length > 1;
    console.warn(
      `Feature flag${t ? "s" : ""} ${e.join(", ")} ${t ? "are" : "is"} not explicitly defined. You are running the esm-bundler build of Vue, which expects these compile-time feature flags to be globally injected via the bundler config in order to get better tree-shaking in the production bundle.

For more details, see https://link.vuejs.org/feature-flags.`
    );
  }
}
const Ee = hc;
function fc(e) {
  return ac(e);
}
function ac(e, t) {
  uc();
  const n = nn();
  n.__VUE__ = !0, process.env.NODE_ENV !== "production" && Ao(n.__VUE_DEVTOOLS_GLOBAL_HOOK__, n);
  const {
    insert: o,
    remove: s,
    patchProp: r,
    createElement: l,
    createText: i,
    createComment: u,
    setText: d,
    setElementText: a,
    parentNode: p,
    nextSibling: g,
    setScopeId: O = ie,
    insertStaticContent: $
  } = e, D = (c, f, h, E = null, v = null, m = null, x = void 0, N = null, y = process.env.NODE_ENV !== "production" && Ge ? !1 : !!f.dynamicChildren) => {
    if (c === f)
      return;
    c && !Ht(c, f) && (E = un(c), ot(c, v, m, !0), c = null), f.patchFlag === -2 && (y = !1, f.dynamicChildren = null);
    const { type: _, ref: A, shapeFlag: w } = f;
    switch (_) {
      case rn:
        Q(c, f, h, E);
        break;
      case we:
        J(c, f, h, E);
        break;
      case En:
        c == null ? K(f, h, E, x) : process.env.NODE_ENV !== "production" && L(c, f, h, x);
        break;
      case Ne:
        I(
          c,
          f,
          h,
          E,
          v,
          m,
          x,
          N,
          y
        );
        break;
      default:
        w & 1 ? ee(
          c,
          f,
          h,
          E,
          v,
          m,
          x,
          N,
          y
        ) : w & 6 ? P(
          c,
          f,
          h,
          E,
          v,
          m,
          x,
          N,
          y
        ) : w & 64 || w & 128 ? _.process(
          c,
          f,
          h,
          E,
          v,
          m,
          x,
          N,
          y,
          Pt
        ) : process.env.NODE_ENV !== "production" && b("Invalid VNode type:", _, `(${typeof _})`);
    }
    A != null && v ? Gt(A, c && c.ref, m, f || c, !f) : A == null && c && c.ref != null && Gt(c.ref, null, m, c, !0);
  }, Q = (c, f, h, E) => {
    if (c == null)
      o(
        f.el = i(f.children),
        h,
        E
      );
    else {
      const v = f.el = c.el;
      f.children !== c.children && d(v, f.children);
    }
  }, J = (c, f, h, E) => {
    c == null ? o(
      f.el = u(f.children || ""),
      h,
      E
    ) : f.el = c.el;
  }, K = (c, f, h, E) => {
    [c.el, c.anchor] = $(
      c.children,
      f,
      h,
      E,
      c.el,
      c.anchor
    );
  }, L = (c, f, h, E) => {
    if (f.children !== c.children) {
      const v = g(c.anchor);
      C(c), [f.el, f.anchor] = $(
        f.children,
        h,
        v,
        E
      );
    } else
      f.el = c.el, f.anchor = c.anchor;
  }, ue = ({ el: c, anchor: f }, h, E) => {
    let v;
    for (; c && c !== f; )
      v = g(c), o(c, h, E), c = v;
    o(f, h, E);
  }, C = ({ el: c, anchor: f }) => {
    let h;
    for (; c && c !== f; )
      h = g(c), s(c), c = h;
    s(f);
  }, ee = (c, f, h, E, v, m, x, N, y) => {
    if (f.type === "svg" ? x = "svg" : f.type === "math" && (x = "mathml"), c == null)
      fe(
        f,
        h,
        E,
        v,
        m,
        x,
        N,
        y
      );
    else {
      const _ = c.el && c.el._isVueCE ? c.el : null;
      try {
        _ && _._beginPatch(), Se(
          c,
          f,
          v,
          m,
          x,
          N,
          y
        );
      } finally {
        _ && _._endPatch();
      }
    }
  }, fe = (c, f, h, E, v, m, x, N) => {
    let y, _;
    const { props: A, shapeFlag: w, transition: S, dirs: R } = c;
    if (y = c.el = l(
      c.type,
      m,
      A && A.is,
      A
    ), w & 8 ? a(y, c.children) : w & 16 && ce(
      c.children,
      y,
      null,
      E,
      v,
      no(c, m),
      x,
      N
    ), R && at(c, null, E, "created"), le(y, c, c.scopeId, x, E), A) {
      for (const z in A)
        z !== "value" && !Bt(z) && r(y, z, null, A[z], m, E);
      "value" in A && r(y, "value", null, A.value, m), (_ = A.onVnodeBeforeMount) && Le(_, E, c);
    }
    process.env.NODE_ENV !== "production" && (On(y, "__vnode", c, !0), On(y, "__vueParentComponent", E, !0)), R && at(c, null, E, "beforeMount");
    const W = pc(v, S);
    W && S.beforeEnter(y), o(y, f, h), ((_ = A && A.onVnodeMounted) || W || R) && Ee(() => {
      _ && Le(_, E, c), W && S.enter(y), R && at(c, null, E, "mounted");
    }, v);
  }, le = (c, f, h, E, v) => {
    if (h && O(c, h), E)
      for (let m = 0; m < E.length; m++)
        O(c, E[m]);
    if (v) {
      let m = v.subTree;
      if (process.env.NODE_ENV !== "production" && m.patchFlag > 0 && m.patchFlag & 2048 && (m = ko(m.children) || m), f === m || jr(m.type) && (m.ssContent === f || m.ssFallback === f)) {
        const x = v.vnode;
        le(
          c,
          x,
          x.scopeId,
          x.slotScopeIds,
          v.parent
        );
      }
    }
  }, ce = (c, f, h, E, v, m, x, N, y = 0) => {
    for (let _ = y; _ < c.length; _++) {
      const A = c[_] = N ? Ze(c[_]) : Ce(c[_]);
      D(
        null,
        A,
        f,
        h,
        E,
        v,
        m,
        x,
        N
      );
    }
  }, Se = (c, f, h, E, v, m, x) => {
    const N = f.el = c.el;
    process.env.NODE_ENV !== "production" && (N.__vnode = f);
    let { patchFlag: y, dynamicChildren: _, dirs: A } = f;
    y |= c.patchFlag & 16;
    const w = c.props || Y, S = f.props || Y;
    let R;
    if (h && pt(h, !1), (R = S.onVnodeBeforeUpdate) && Le(R, h, f, c), A && at(f, c, h, "beforeUpdate"), h && pt(h, !0), process.env.NODE_ENV !== "production" && Ge && (y = 0, x = !1, _ = null), (w.innerHTML && S.innerHTML == null || w.textContent && S.textContent == null) && a(N, ""), _ ? (De(
      c.dynamicChildren,
      _,
      N,
      h,
      E,
      no(f, v),
      m
    ), process.env.NODE_ENV !== "production" && _n(c, f)) : x || ge(
      c,
      f,
      N,
      null,
      h,
      E,
      no(f, v),
      m,
      !1
    ), y > 0) {
      if (y & 16)
        V(N, w, S, h, v);
      else if (y & 2 && w.class !== S.class && r(N, "class", null, S.class, v), y & 4 && r(N, "style", w.style, S.style, v), y & 8) {
        const W = f.dynamicProps;
        for (let z = 0; z < W.length; z++) {
          const q = W[z], ve = w[q], me = S[q];
          (me !== ve || q === "value") && r(N, q, ve, me, v, h);
        }
      }
      y & 1 && c.children !== f.children && a(N, f.children);
    } else !x && _ == null && V(N, w, S, h, v);
    ((R = S.onVnodeUpdated) || A) && Ee(() => {
      R && Le(R, h, f, c), A && at(f, c, h, "updated");
    }, E);
  }, De = (c, f, h, E, v, m, x) => {
    for (let N = 0; N < f.length; N++) {
      const y = c[N], _ = f[N], A = (
        // oldVNode may be an errored async setup() component inside Suspense
        // which will not have a mounted element
        y.el && // - In the case of a Fragment, we need to provide the actual parent
        // of the Fragment itself so it can move its children.
        (y.type === Ne || // - In the case of different nodes, there is going to be a replacement
        // which also requires the correct parent container
        !Ht(y, _) || // - In the case of a component, it could contain anything.
        y.shapeFlag & 198) ? p(y.el) : (
          // In other cases, the parent container is not actually used so we
          // just pass the block element here to avoid a DOM parentNode call.
          h
        )
      );
      D(
        y,
        _,
        A,
        null,
        E,
        v,
        m,
        x,
        !0
      );
    }
  }, V = (c, f, h, E, v) => {
    if (f !== h) {
      if (f !== Y)
        for (const m in f)
          !Bt(m) && !(m in h) && r(
            c,
            m,
            f[m],
            null,
            v,
            E
          );
      for (const m in h) {
        if (Bt(m)) continue;
        const x = h[m], N = f[m];
        x !== N && m !== "value" && r(c, m, N, x, v, E);
      }
      "value" in h && r(c, "value", f.value, h.value, v);
    }
  }, I = (c, f, h, E, v, m, x, N, y) => {
    const _ = f.el = c ? c.el : i(""), A = f.anchor = c ? c.anchor : i("");
    let { patchFlag: w, dynamicChildren: S, slotScopeIds: R } = f;
    process.env.NODE_ENV !== "production" && // #5523 dev root fragment may inherit directives
    (Ge || w & 2048) && (w = 0, y = !1, S = null), R && (N = N ? N.concat(R) : R), c == null ? (o(_, h, E), o(A, h, E), ce(
      // #10007
      // such fragment like `<></>` will be compiled into
      // a fragment which doesn't have a children.
      // In this case fallback to an empty array
      f.children || [],
      h,
      A,
      v,
      m,
      x,
      N,
      y
    )) : w > 0 && w & 64 && S && // #2715 the previous fragment could've been a BAILed one as a result
    // of renderSlot() with no valid children
    c.dynamicChildren && c.dynamicChildren.length === S.length ? (De(
      c.dynamicChildren,
      S,
      h,
      v,
      m,
      x,
      N
    ), process.env.NODE_ENV !== "production" ? _n(c, f) : (
      // #2080 if the stable fragment has a key, it's a <template v-for> that may
      //  get moved around. Make sure all root level vnodes inherit el.
      // #2134 or if it's a component root, it may also get moved around
      // as the component is being moved.
      (f.key != null || v && f === v.subTree) && _n(
        c,
        f,
        !0
        /* shallow */
      )
    )) : ge(
      c,
      f,
      h,
      A,
      v,
      m,
      x,
      N,
      y
    );
  }, P = (c, f, h, E, v, m, x, N, y) => {
    f.slotScopeIds = N, c == null ? f.shapeFlag & 512 ? v.ctx.activate(
      f,
      h,
      E,
      x,
      y
    ) : k(
      f,
      h,
      E,
      v,
      m,
      x,
      y
    ) : oe(c, f, y);
  }, k = (c, f, h, E, v, m, x) => {
    const N = c.component = Oc(
      c,
      E,
      v
    );
    if (process.env.NODE_ENV !== "production" && N.type.__hmrId && Zi(N), process.env.NODE_ENV !== "production" && (hn(c), bt(N, "mount")), Fo(c) && (N.ctx.renderer = Pt), process.env.NODE_ENV !== "production" && bt(N, "init"), wc(N, !1, x), process.env.NODE_ENV !== "production" && Ot(N, "init"), process.env.NODE_ENV !== "production" && Ge && (c.el = null), N.asyncDep) {
      if (v && v.registerDep(N, H, x), !c.el) {
        const y = N.subTree = xe(we);
        J(null, y, f, h), c.placeholder = y.el;
      }
    } else
      H(
        N,
        c,
        f,
        h,
        v,
        m,
        x
      );
    process.env.NODE_ENV !== "production" && (gn(), Ot(N, "mount"));
  }, oe = (c, f, h) => {
    const E = f.component = c.component;
    if (Yl(c, f, h))
      if (E.asyncDep && !E.asyncResolved) {
        process.env.NODE_ENV !== "production" && hn(f), F(E, f, h), process.env.NODE_ENV !== "production" && gn();
        return;
      } else
        E.next = f, E.update();
    else
      f.el = c.el, E.vnode = f;
  }, H = (c, f, h, E, v, m, x) => {
    const N = () => {
      if (c.isMounted) {
        let { next: w, bu: S, u: R, parent: W, vnode: z } = c;
        {
          const He = Rr(c);
          if (He) {
            w && (w.el = z.el, F(c, w, x)), He.asyncDep.then(() => {
              Ee(() => {
                c.isUnmounted || _();
              }, v);
            });
            return;
          }
        }
        let q = w, ve;
        process.env.NODE_ENV !== "production" && hn(w || c.vnode), pt(c, !1), w ? (w.el = z.el, F(c, w, x)) : w = z, S && Mt(S), (ve = w.props && w.props.onVnodeBeforeUpdate) && Le(ve, W, w, z), pt(c, !0), process.env.NODE_ENV !== "production" && bt(c, "render");
        const me = us(c);
        process.env.NODE_ENV !== "production" && Ot(c, "render");
        const je = c.subTree;
        c.subTree = me, process.env.NODE_ENV !== "production" && bt(c, "patch"), D(
          je,
          me,
          // parent may have changed if it's in a teleport
          p(je.el),
          // anchor may have changed if it's in a fragment
          un(je),
          c,
          v,
          m
        ), process.env.NODE_ENV !== "production" && Ot(c, "patch"), w.el = me.el, q === null && Jl(c, me.el), R && Ee(R, v), (ve = w.props && w.props.onVnodeUpdated) && Ee(
          () => Le(ve, W, w, z),
          v
        ), process.env.NODE_ENV !== "production" && pr(c), process.env.NODE_ENV !== "production" && gn();
      } else {
        let w;
        const { el: S, props: R } = f, { bm: W, m: z, parent: q, root: ve, type: me } = c, je = qt(f);
        pt(c, !1), W && Mt(W), !je && (w = R && R.onVnodeBeforeMount) && Le(w, q, f), pt(c, !0);
        {
          ve.ce && ve.ce._hasShadowRoot() && ve.ce._injectChildStyle(me), process.env.NODE_ENV !== "production" && bt(c, "render");
          const He = c.subTree = us(c);
          process.env.NODE_ENV !== "production" && Ot(c, "render"), process.env.NODE_ENV !== "production" && bt(c, "patch"), D(
            null,
            He,
            h,
            E,
            c,
            v,
            m
          ), process.env.NODE_ENV !== "production" && Ot(c, "patch"), f.el = He.el;
        }
        if (z && Ee(z, v), !je && (w = R && R.onVnodeMounted)) {
          const He = f;
          Ee(
            () => Le(w, q, He),
            v
          );
        }
        (f.shapeFlag & 256 || q && qt(q.vnode) && q.vnode.shapeFlag & 256) && c.a && Ee(c.a, v), c.isMounted = !0, process.env.NODE_ENV !== "production" && rl(c), f = h = E = null;
      }
    };
    c.scope.on();
    const y = c.effect = new Ls(N);
    c.scope.off();
    const _ = c.update = y.run.bind(y), A = c.job = y.runIfDirty.bind(y);
    A.i = c, A.id = c.uid, y.scheduler = () => Ln(A), pt(c, !0), process.env.NODE_ENV !== "production" && (y.onTrack = c.rtc ? (w) => Mt(c.rtc, w) : void 0, y.onTrigger = c.rtg ? (w) => Mt(c.rtg, w) : void 0), _();
  }, F = (c, f, h) => {
    f.component = c;
    const E = c.vnode.props;
    c.vnode = f, c.next = null, Xl(c, f.props, E, h), cc(c, f.children, h), Ae(), es(c), Me();
  }, ge = (c, f, h, E, v, m, x, N, y = !1) => {
    const _ = c && c.children, A = c ? c.shapeFlag : 0, w = f.children, { patchFlag: S, shapeFlag: R } = f;
    if (S > 0) {
      if (S & 128) {
        $t(
          _,
          w,
          h,
          E,
          v,
          m,
          x,
          N,
          y
        );
        return;
      } else if (S & 256) {
        Un(
          _,
          w,
          h,
          E,
          v,
          m,
          x,
          N,
          y
        );
        return;
      }
    }
    R & 8 ? (A & 16 && It(_, v, m), w !== _ && a(h, w)) : A & 16 ? R & 16 ? $t(
      _,
      w,
      h,
      E,
      v,
      m,
      x,
      N,
      y
    ) : It(_, v, m, !0) : (A & 8 && a(h, ""), R & 16 && ce(
      w,
      h,
      E,
      v,
      m,
      x,
      N,
      y
    ));
  }, Un = (c, f, h, E, v, m, x, N, y) => {
    c = c || wt, f = f || wt;
    const _ = c.length, A = f.length, w = Math.min(_, A);
    let S;
    for (S = 0; S < w; S++) {
      const R = f[S] = y ? Ze(f[S]) : Ce(f[S]);
      D(
        c[S],
        R,
        h,
        null,
        v,
        m,
        x,
        N,
        y
      );
    }
    _ > A ? It(
      c,
      v,
      m,
      !0,
      !1,
      w
    ) : ce(
      f,
      h,
      E,
      v,
      m,
      x,
      N,
      y,
      w
    );
  }, $t = (c, f, h, E, v, m, x, N, y) => {
    let _ = 0;
    const A = f.length;
    let w = c.length - 1, S = A - 1;
    for (; _ <= w && _ <= S; ) {
      const R = c[_], W = f[_] = y ? Ze(f[_]) : Ce(f[_]);
      if (Ht(R, W))
        D(
          R,
          W,
          h,
          null,
          v,
          m,
          x,
          N,
          y
        );
      else
        break;
      _++;
    }
    for (; _ <= w && _ <= S; ) {
      const R = c[w], W = f[S] = y ? Ze(f[S]) : Ce(f[S]);
      if (Ht(R, W))
        D(
          R,
          W,
          h,
          null,
          v,
          m,
          x,
          N,
          y
        );
      else
        break;
      w--, S--;
    }
    if (_ > w) {
      if (_ <= S) {
        const R = S + 1, W = R < A ? f[R].el : E;
        for (; _ <= S; )
          D(
            null,
            f[_] = y ? Ze(f[_]) : Ce(f[_]),
            h,
            W,
            v,
            m,
            x,
            N,
            y
          ), _++;
      }
    } else if (_ > S)
      for (; _ <= w; )
        ot(c[_], v, m, !0), _++;
    else {
      const R = _, W = _, z = /* @__PURE__ */ new Map();
      for (_ = W; _ <= S; _++) {
        const ae = f[_] = y ? Ze(f[_]) : Ce(f[_]);
        ae.key != null && (process.env.NODE_ENV !== "production" && z.has(ae.key) && b(
          "Duplicate keys found during update:",
          JSON.stringify(ae.key),
          "Make sure keys are unique."
        ), z.set(ae.key, _));
      }
      let q, ve = 0;
      const me = S - W + 1;
      let je = !1, He = 0;
      const At = new Array(me);
      for (_ = 0; _ < me; _++) At[_] = 0;
      for (_ = R; _ <= w; _++) {
        const ae = c[_];
        if (ve >= me) {
          ot(ae, v, m, !0);
          continue;
        }
        let ke;
        if (ae.key != null)
          ke = z.get(ae.key);
        else
          for (q = W; q <= S; q++)
            if (At[q - W] === 0 && Ht(ae, f[q])) {
              ke = q;
              break;
            }
        ke === void 0 ? ot(ae, v, m, !0) : (At[ke - W] = _ + 1, ke >= He ? He = ke : je = !0, D(
          ae,
          f[ke],
          h,
          null,
          v,
          m,
          x,
          N,
          y
        ), ve++);
      }
      const Go = je ? dc(At) : wt;
      for (q = Go.length - 1, _ = me - 1; _ >= 0; _--) {
        const ae = W + _, ke = f[ae], qo = f[ae + 1], Yo = ae + 1 < A ? (
          // #13559, #14173 fallback to el placeholder for unresolved async component
          qo.el || Fr(qo)
        ) : E;
        At[_] === 0 ? D(
          null,
          ke,
          h,
          Yo,
          v,
          m,
          x,
          N,
          y
        ) : je && (q < 0 || _ !== Go[q] ? yt(ke, h, Yo, 2) : q--);
      }
    }
  }, yt = (c, f, h, E, v = null) => {
    const { el: m, type: x, transition: N, children: y, shapeFlag: _ } = c;
    if (_ & 6) {
      yt(c.component.subTree, f, h, E);
      return;
    }
    if (_ & 128) {
      c.suspense.move(f, h, E);
      return;
    }
    if (_ & 64) {
      x.move(c, f, h, Pt);
      return;
    }
    if (x === Ne) {
      o(m, f, h);
      for (let w = 0; w < y.length; w++)
        yt(y[w], f, h, E);
      o(c.anchor, f, h);
      return;
    }
    if (x === En) {
      ue(c, f, h);
      return;
    }
    if (E !== 2 && _ & 1 && N)
      if (E === 0)
        N.beforeEnter(m), o(m, f, h), Ee(() => N.enter(m), v);
      else {
        const { leave: w, delayLeave: S, afterLeave: R } = N, W = () => {
          c.ctx.isUnmounted ? s(m) : o(m, f, h);
        }, z = () => {
          m._isLeaving && m[_l](
            !0
            /* cancelled */
          ), w(m, () => {
            W(), R && R();
          });
        };
        S ? S(m, W, z) : z();
      }
    else
      o(m, f, h);
  }, ot = (c, f, h, E = !1, v = !1) => {
    const {
      type: m,
      props: x,
      ref: N,
      children: y,
      dynamicChildren: _,
      shapeFlag: A,
      patchFlag: w,
      dirs: S,
      cacheIndex: R
    } = c;
    if (w === -2 && (v = !1), N != null && (Ae(), Gt(N, null, h, c, !0), Me()), R != null && (f.renderCache[R] = void 0), A & 256) {
      f.ctx.deactivate(c);
      return;
    }
    const W = A & 1 && S, z = !qt(c);
    let q;
    if (z && (q = x && x.onVnodeBeforeUnmount) && Le(q, f, c), A & 6)
      zr(c.component, h, E);
    else {
      if (A & 128) {
        c.suspense.unmount(h, E);
        return;
      }
      W && at(c, null, f, "beforeUnmount"), A & 64 ? c.type.remove(
        c,
        f,
        h,
        Pt,
        E
      ) : _ && // #5154
      // when v-once is used inside a block, setBlockTracking(-1) marks the
      // parent block with hasOnce: true
      // so that it doesn't take the fast path during unmount - otherwise
      // components nested in v-once are never unmounted.
      !_.hasOnce && // #1153: fast path should not be taken for non-stable (v-for) fragments
      (m !== Ne || w > 0 && w & 64) ? It(
        _,
        f,
        h,
        !1,
        !0
      ) : (m === Ne && w & 384 || !v && A & 16) && It(y, f, h), E && Kn(c);
    }
    (z && (q = x && x.onVnodeUnmounted) || W) && Ee(() => {
      q && Le(q, f, c), W && at(c, null, f, "unmounted");
    }, h);
  }, Kn = (c) => {
    const { type: f, el: h, anchor: E, transition: v } = c;
    if (f === Ne) {
      process.env.NODE_ENV !== "production" && c.patchFlag > 0 && c.patchFlag & 2048 && v && !v.persisted ? c.children.forEach((x) => {
        x.type === we ? s(x.el) : Kn(x);
      }) : Jr(h, E);
      return;
    }
    if (f === En) {
      C(c);
      return;
    }
    const m = () => {
      s(h), v && !v.persisted && v.afterLeave && v.afterLeave();
    };
    if (c.shapeFlag & 1 && v && !v.persisted) {
      const { leave: x, delayLeave: N } = v, y = () => x(h, m);
      N ? N(c.el, m, y) : y();
    } else
      m();
  }, Jr = (c, f) => {
    let h;
    for (; c !== f; )
      h = g(c), s(c), c = h;
    s(f);
  }, zr = (c, f, h) => {
    process.env.NODE_ENV !== "production" && c.type.__hmrId && el(c);
    const { bum: E, scope: v, job: m, subTree: x, um: N, m: y, a: _ } = c;
    gs(y), gs(_), E && Mt(E), v.stop(), m && (m.flags |= 8, ot(x, c, f, h)), N && Ee(N, f), Ee(() => {
      c.isUnmounted = !0;
    }, f), process.env.NODE_ENV !== "production" && ll(c);
  }, It = (c, f, h, E = !1, v = !1, m = 0) => {
    for (let x = m; x < c.length; x++)
      ot(c[x], f, h, E, v);
  }, un = (c) => {
    if (c.shapeFlag & 6)
      return un(c.component.subTree);
    if (c.shapeFlag & 128)
      return c.suspense.next();
    const f = g(c.anchor || c.el), h = f && f[vl];
    return h ? g(h) : f;
  };
  let Gn = !1;
  const Ko = (c, f, h) => {
    let E;
    c == null ? f._vnode && (ot(f._vnode, null, null, !0), E = f._vnode.component) : D(
      f._vnode || null,
      c,
      f,
      null,
      null,
      null,
      h
    ), f._vnode = c, Gn || (Gn = !0, es(E), ur(), Gn = !1);
  }, Pt = {
    p: D,
    um: ot,
    m: yt,
    r: Kn,
    mt: k,
    mc: ce,
    pc: ge,
    pbc: De,
    n: un,
    o: e
  };
  return {
    render: Ko,
    hydrate: void 0,
    createApp: Wl(Ko)
  };
}
function no({ type: e, props: t }, n) {
  return n === "svg" && e === "foreignObject" || n === "mathml" && e === "annotation-xml" && t && t.encoding && t.encoding.includes("html") ? void 0 : n;
}
function pt({ effect: e, job: t }, n) {
  n ? (e.flags |= 32, t.flags |= 4) : (e.flags &= -33, t.flags &= -5);
}
function pc(e, t) {
  return (!e || e && !e.pendingBranch) && t && !t.persisted;
}
function _n(e, t, n = !1) {
  const o = e.children, s = t.children;
  if (T(o) && T(s))
    for (let r = 0; r < o.length; r++) {
      const l = o[r];
      let i = s[r];
      i.shapeFlag & 1 && !i.dynamicChildren && ((i.patchFlag <= 0 || i.patchFlag === 32) && (i = s[r] = Ze(s[r]), i.el = l.el), !n && i.patchFlag !== -2 && _n(l, i)), i.type === rn && (i.patchFlag === -1 && (i = s[r] = Ze(i)), i.el = l.el), i.type === we && !i.el && (i.el = l.el), process.env.NODE_ENV !== "production" && i.el && (i.el.__vnode = i);
    }
}
function dc(e) {
  const t = e.slice(), n = [0];
  let o, s, r, l, i;
  const u = e.length;
  for (o = 0; o < u; o++) {
    const d = e[o];
    if (d !== 0) {
      if (s = n[n.length - 1], e[s] < d) {
        t[o] = s, n.push(o);
        continue;
      }
      for (r = 0, l = n.length - 1; r < l; )
        i = r + l >> 1, e[n[i]] < d ? r = i + 1 : l = i;
      d < e[n[r]] && (r > 0 && (t[o] = n[r - 1]), n[r] = o);
    }
  }
  for (r = n.length, l = n[r - 1]; r-- > 0; )
    n[r] = l, l = t[l];
  return n;
}
function Rr(e) {
  const t = e.subTree.component;
  if (t)
    return t.asyncDep && !t.asyncResolved ? t : Rr(t);
}
function gs(e) {
  if (e)
    for (let t = 0; t < e.length; t++)
      e[t].flags |= 8;
}
function Fr(e) {
  if (e.placeholder)
    return e.placeholder;
  const t = e.component;
  return t ? Fr(t.subTree) : null;
}
const jr = (e) => e.__isSuspense;
function hc(e, t) {
  t && t.pendingBranch ? T(e) ? t.effects.push(...e) : t.effects.push(e) : cr(e);
}
const Ne = /* @__PURE__ */ Symbol.for("v-fgt"), rn = /* @__PURE__ */ Symbol.for("v-txt"), we = /* @__PURE__ */ Symbol.for("v-cmt"), En = /* @__PURE__ */ Symbol.for("v-stc"), Yt = [];
let Oe = null;
function We(e = !1) {
  Yt.push(Oe = e ? null : []);
}
function gc() {
  Yt.pop(), Oe = Yt[Yt.length - 1] || null;
}
let Xt = 1;
function Pn(e, t = !1) {
  Xt += e, e < 0 && Oe && t && (Oe.hasOnce = !0);
}
function Hr(e) {
  return e.dynamicChildren = Xt > 0 ? Oe || wt : null, gc(), Xt > 0 && Oe && Oe.push(e), e;
}
function ze(e, t, n, o, s, r) {
  return Hr(
    te(
      e,
      t,
      n,
      o,
      s,
      r,
      !0
    )
  );
}
function vc(e, t, n, o, s) {
  return Hr(
    xe(
      e,
      t,
      n,
      o,
      s,
      !0
    )
  );
}
function Ct(e) {
  return e ? e.__v_isVNode === !0 : !1;
}
function Ht(e, t) {
  if (process.env.NODE_ENV !== "production" && t.shapeFlag & 6 && e.component) {
    const n = vn.get(t.type);
    if (n && n.has(e.component))
      return e.shapeFlag &= -257, t.shapeFlag &= -513, !1;
  }
  return e.type === t.type && e.key === t.key;
}
const mc = (...e) => Lr(
  ...e
), kr = ({ key: e }) => e ?? null, yn = ({
  ref: e,
  ref_key: t,
  ref_for: n
}) => (typeof e == "number" && (e = "" + e), e != null ? X(e) || /* @__PURE__ */ ne(e) || M(e) ? { i: be, r: e, k: t, f: !!n } : e : null);
function te(e, t = null, n = null, o = 0, s = null, r = e === Ne ? 0 : 1, l = !1, i = !1) {
  const u = {
    __v_isVNode: !0,
    __v_skip: !0,
    type: e,
    props: t,
    key: t && kr(t),
    ref: t && yn(t),
    scopeId: hr,
    slotScopeIds: null,
    children: n,
    component: null,
    suspense: null,
    ssContent: null,
    ssFallback: null,
    dirs: null,
    transition: null,
    el: null,
    anchor: null,
    target: null,
    targetStart: null,
    targetAnchor: null,
    staticCount: 0,
    shapeFlag: r,
    patchFlag: o,
    dynamicProps: s,
    dynamicChildren: null,
    appContext: null,
    ctx: be
  };
  return i ? (Bo(u, n), r & 128 && e.normalize(u)) : n && (u.shapeFlag |= X(n) ? 8 : 16), process.env.NODE_ENV !== "production" && u.key !== u.key && b("VNode created with invalid key (NaN). VNode type:", u.type), Xt > 0 && // avoid a block node from tracking itself
  !l && // has current parent block
  Oe && // presence of a patch flag indicates this node needs patching on updates.
  // component nodes also should always be patched, because even if the
  // component doesn't need to update, it needs to persist the instance on to
  // the next vnode so that it can be properly unmounted later.
  (u.patchFlag > 0 || r & 6) && // the EVENTS flag is only for hydration and if it is the only flag, the
  // vnode should not be considered dynamic due to handler caching.
  u.patchFlag !== 32 && Oe.push(u), u;
}
const xe = process.env.NODE_ENV !== "production" ? mc : Lr;
function Lr(e, t = null, n = null, o = 0, s = null, r = !1) {
  if ((!e || e === $l) && (process.env.NODE_ENV !== "production" && !e && b(`Invalid vnode type when creating vnode: ${e}.`), e = we), Ct(e)) {
    const i = ft(
      e,
      t,
      !0
      /* mergeRef: true */
    );
    return n && Bo(i, n), Xt > 0 && !r && Oe && (i.shapeFlag & 6 ? Oe[Oe.indexOf(e)] = i : Oe.push(i)), i.patchFlag = -2, i;
  }
  if (qr(e) && (e = e.__vccOpts), t) {
    t = _c(t);
    let { class: i, style: u } = t;
    i && !X(i) && (t.class = Fn(i)), U(u) && (/* @__PURE__ */ xn(u) && !T(u) && (u = Z({}, u)), t.style = wo(u));
  }
  const l = X(e) ? 1 : jr(e) ? 128 : ml(e) ? 64 : U(e) ? 4 : M(e) ? 2 : 0;
  return process.env.NODE_ENV !== "production" && l & 4 && /* @__PURE__ */ xn(e) && (e = /* @__PURE__ */ j(e), b(
    "Vue received a Component that was made a reactive object. This can lead to unnecessary performance overhead and should be avoided by marking the component with `markRaw` or using `shallowRef` instead of `ref`.",
    `
Component that was made reactive: `,
    e
  )), te(
    e,
    t,
    n,
    o,
    s,
    l,
    r,
    !0
  );
}
function _c(e) {
  return e ? /* @__PURE__ */ xn(e) || Tr(e) ? Z({}, e) : e : null;
}
function ft(e, t, n = !1, o = !1) {
  const { props: s, ref: r, patchFlag: l, children: i, transition: u } = e, d = t ? yc(s || {}, t) : s, a = {
    __v_isVNode: !0,
    __v_skip: !0,
    type: e.type,
    props: d,
    key: d && kr(d),
    ref: t && t.ref ? (
      // #2078 in the case of <component :is="vnode" ref="extra"/>
      // if the vnode itself already has a ref, cloneVNode will need to merge
      // the refs so the single vnode can be set on multiple refs
      n && r ? T(r) ? r.concat(yn(t)) : [r, yn(t)] : yn(t)
    ) : r,
    scopeId: e.scopeId,
    slotScopeIds: e.slotScopeIds,
    children: process.env.NODE_ENV !== "production" && l === -1 && T(i) ? i.map(Wr) : i,
    target: e.target,
    targetStart: e.targetStart,
    targetAnchor: e.targetAnchor,
    staticCount: e.staticCount,
    shapeFlag: e.shapeFlag,
    // if the vnode is cloned with extra props, we can no longer assume its
    // existing patch flag to be reliable and need to add the FULL_PROPS flag.
    // note: preserve flag for fragments since they use the flag for children
    // fast paths only.
    patchFlag: t && e.type !== Ne ? l === -1 ? 16 : l | 16 : l,
    dynamicProps: e.dynamicProps,
    dynamicChildren: e.dynamicChildren,
    appContext: e.appContext,
    dirs: e.dirs,
    transition: u,
    // These should technically only be non-null on mounted VNodes. However,
    // they *should* be copied for kept-alive vnodes. So we just always copy
    // them since them being non-null during a mount doesn't affect the logic as
    // they will simply be overwritten.
    component: e.component,
    suspense: e.suspense,
    ssContent: e.ssContent && ft(e.ssContent),
    ssFallback: e.ssFallback && ft(e.ssFallback),
    placeholder: e.placeholder,
    el: e.el,
    anchor: e.anchor,
    ctx: e.ctx,
    ce: e.ce
  };
  return u && o && Ro(
    a,
    u.clone(a)
  ), a;
}
function Wr(e) {
  const t = ft(e);
  return T(e.children) && (t.children = e.children.map(Wr)), t;
}
function Ec(e = " ", t = 0) {
  return xe(rn, null, e, t);
}
function vs(e = "", t = !1) {
  return t ? (We(), vc(we, null, e)) : xe(we, null, e);
}
function Ce(e) {
  return e == null || typeof e == "boolean" ? xe(we) : T(e) ? xe(
    Ne,
    null,
    // #3666, avoid reference pollution when reusing vnode
    e.slice()
  ) : Ct(e) ? Ze(e) : xe(rn, null, String(e));
}
function Ze(e) {
  return e.el === null && e.patchFlag !== -1 || e.memo ? e : ft(e);
}
function Bo(e, t) {
  let n = 0;
  const { shapeFlag: o } = e;
  if (t == null)
    t = null;
  else if (T(t))
    n = 16;
  else if (typeof t == "object")
    if (o & 65) {
      const s = t.default;
      s && (s._c && (s._d = !1), Bo(e, s()), s._c && (s._d = !0));
      return;
    } else {
      n = 32;
      const s = t._;
      !s && !Tr(t) ? t._ctx = be : s === 3 && be && (be.slots._ === 1 ? t._ = 1 : (t._ = 2, e.patchFlag |= 1024));
    }
  else M(t) ? (t = { default: t, _ctx: be }, n = 32) : (t = String(t), o & 64 ? (n = 16, t = [Ec(t)]) : n = 8);
  e.children = t, e.shapeFlag |= n;
}
function yc(...e) {
  const t = {};
  for (let n = 0; n < e.length; n++) {
    const o = e[n];
    for (const s in o)
      if (s === "class")
        t.class !== o.class && (t.class = Fn([t.class, o.class]));
      else if (s === "style")
        t.style = wo([t.style, o.style]);
      else if (en(s)) {
        const r = t[s], l = o[s];
        l && r !== l && !(T(r) && r.includes(l)) && (t[s] = r ? [].concat(r, l) : l);
      } else s !== "" && (t[s] = o[s]);
  }
  return t;
}
function Le(e, t, n, o = null) {
  Ye(e, t, 7, [
    n,
    o
  ]);
}
const Nc = xr();
let bc = 0;
function Oc(e, t, n) {
  const o = e.type, s = (t ? t.appContext : e.appContext) || Nc, r = {
    uid: bc++,
    vnode: e,
    type: o,
    parent: t,
    appContext: s,
    root: null,
    // to be immediately set
    next: null,
    subTree: null,
    // will be set synchronously right after creation
    effect: null,
    update: null,
    // will be set synchronously right after creation
    job: null,
    scope: new vi(
      !0
      /* detached */
    ),
    render: null,
    proxy: null,
    exposed: null,
    exposeProxy: null,
    withProxy: null,
    provides: t ? t.provides : Object.create(s.provides),
    ids: t ? t.ids : ["", 0, 0],
    accessCache: null,
    renderCache: [],
    // local resolved assets
    components: null,
    directives: null,
    // resolved props and emits options
    propsOptions: Ir(o, s),
    emitsOptions: wr(o, s),
    // emit
    emit: null,
    // to be set immediately
    emitted: null,
    // props default value
    propsDefaults: Y,
    // inheritAttrs
    inheritAttrs: o.inheritAttrs,
    // state
    ctx: Y,
    data: Y,
    props: Y,
    attrs: Y,
    slots: Y,
    refs: Y,
    setupState: Y,
    setupContext: null,
    // suspense related
    suspense: n,
    suspenseId: n ? n.pendingId : 0,
    asyncDep: null,
    asyncResolved: !1,
    // lifecycle hooks
    // not using enums here because it results in computed properties
    isMounted: !1,
    isUnmounted: !1,
    isDeactivated: !1,
    bc: null,
    c: null,
    bm: null,
    m: null,
    bu: null,
    u: null,
    um: null,
    bum: null,
    da: null,
    a: null,
    rtg: null,
    rtc: null,
    ec: null,
    sp: null
  };
  return process.env.NODE_ENV !== "production" ? r.ctx = Il(r) : r.ctx = { _: r }, r.root = t ? t.root : r, r.emit = Ul.bind(null, r), e.ce && e.ce(r), r;
}
let se = null;
const Br = () => se || be;
let An, _o;
{
  const e = nn(), t = (n, o) => {
    let s;
    return (s = e[n]) || (s = e[n] = []), s.push(o), (r) => {
      s.length > 1 ? s.forEach((l) => l(r)) : s[0](r);
    };
  };
  An = t(
    "__VUE_INSTANCE_SETTERS__",
    (n) => se = n
  ), _o = t(
    "__VUE_SSR_SETTERS__",
    (n) => Zt = n
  );
}
const ln = (e) => {
  const t = se;
  return An(e), e.scope.on(), () => {
    e.scope.off(), An(t);
  };
}, ms = () => {
  se && se.scope.off(), An(null);
}, xc = /* @__PURE__ */ tt("slot,component");
function Eo(e, { isNativeTag: t }) {
  (xc(e) || t(e)) && b(
    "Do not use built-in or reserved HTML elements as component id: " + e
  );
}
function Ur(e) {
  return e.vnode.shapeFlag & 4;
}
let Zt = !1;
function wc(e, t = !1, n = !1) {
  t && _o(t);
  const { props: o, children: s } = e.vnode, r = Ur(e);
  zl(e, o, r, t), lc(e, s, n || t);
  const l = r ? Dc(e, t) : void 0;
  return t && _o(!1), l;
}
function Dc(e, t) {
  const n = e.type;
  if (process.env.NODE_ENV !== "production") {
    if (n.name && Eo(n.name, e.appContext.config), n.components) {
      const s = Object.keys(n.components);
      for (let r = 0; r < s.length; r++)
        Eo(s[r], e.appContext.config);
    }
    if (n.directives) {
      const s = Object.keys(n.directives);
      for (let r = 0; r < s.length; r++)
        gr(s[r]);
    }
    n.compilerOptions && Vc() && b(
      '"compilerOptions" is only supported when using a build of Vue that includes the runtime compiler. Since you are using a runtime-only build, the options should be passed via your build tool config instead.'
    );
  }
  e.accessCache = /* @__PURE__ */ Object.create(null), e.proxy = new Proxy(e.ctx, Nr), process.env.NODE_ENV !== "production" && Pl(e);
  const { setup: o } = n;
  if (o) {
    Ae();
    const s = e.setupContext = o.length > 1 ? Cc(e) : null, r = ln(e), l = Tt(
      o,
      e,
      0,
      [
        process.env.NODE_ENV !== "production" ? /* @__PURE__ */ Ke(e.props) : e.props,
        s
      ]
    ), i = bo(l);
    if (Me(), r(), (i || e.sp) && !qt(e) && _r(e), i) {
      if (l.then(ms, ms), t)
        return l.then((u) => {
          _s(e, u, t);
        }).catch((u) => {
          on(u, e, 0);
        });
      if (e.asyncDep = l, process.env.NODE_ENV !== "production" && !e.suspense) {
        const u = cn(e, n);
        b(
          `Component <${u}>: setup function returned a promise, but no <Suspense> boundary was found in the parent component tree. A component with async setup() must be nested in a <Suspense> in order to be rendered.`
        );
      }
    } else
      _s(e, l, t);
  } else
    Kr(e, t);
}
function _s(e, t, n) {
  M(t) ? e.type.__ssrInlineRender ? e.ssrRender = t : e.render = t : U(t) ? (process.env.NODE_ENV !== "production" && Ct(t) && b(
    "setup() should not return VNodes directly - return a render function instead."
  ), process.env.NODE_ENV !== "production" && (e.devtoolsRawSetupState = t), e.setupState = sr(t), process.env.NODE_ENV !== "production" && Al(e)) : process.env.NODE_ENV !== "production" && t !== void 0 && b(
    `setup() should return an object. Received: ${t === null ? "null" : typeof t}`
  ), Kr(e, n);
}
const Vc = () => !0;
function Kr(e, t, n) {
  const o = e.type;
  e.render || (e.render = o.render || ie);
  {
    const s = ln(e);
    Ae();
    try {
      Rl(e);
    } finally {
      Me(), s();
    }
  }
  process.env.NODE_ENV !== "production" && !o.render && e.render === ie && !t && (o.template ? b(
    'Component provided template option but runtime compilation is not supported in this build of Vue. Configure your bundler to alias "vue" to "vue/dist/vue.esm-bundler.js".'
  ) : b("Component is missing template or render function: ", o));
}
const Es = process.env.NODE_ENV !== "production" ? {
  get(e, t) {
    return $n(), re(e, "get", ""), e[t];
  },
  set() {
    return b("setupContext.attrs is readonly."), !1;
  },
  deleteProperty() {
    return b("setupContext.attrs is readonly."), !1;
  }
} : {
  get(e, t) {
    return re(e, "get", ""), e[t];
  }
};
function Sc(e) {
  return new Proxy(e.slots, {
    get(t, n) {
      return re(e, "get", "$slots"), t[n];
    }
  });
}
function Cc(e) {
  const t = (n) => {
    if (process.env.NODE_ENV !== "production" && (e.exposed && b("expose() should be called only once per setup()."), n != null)) {
      let o = typeof n;
      o === "object" && (T(n) ? o = "array" : /* @__PURE__ */ ne(n) && (o = "ref")), o !== "object" && b(
        `expose() should be passed a plain object, received ${o}.`
      );
    }
    e.exposed = n || {};
  };
  if (process.env.NODE_ENV !== "production") {
    let n, o;
    return Object.freeze({
      get attrs() {
        return n || (n = new Proxy(e.attrs, Es));
      },
      get slots() {
        return o || (o = Sc(e));
      },
      get emit() {
        return (s, ...r) => e.emit(s, ...r);
      },
      expose: t
    });
  } else
    return {
      attrs: new Proxy(e.attrs, Es),
      slots: e.slots,
      emit: e.emit,
      expose: t
    };
}
function Uo(e) {
  return e.exposed ? e.exposeProxy || (e.exposeProxy = new Proxy(sr(Fi(e.exposed)), {
    get(t, n) {
      if (n in t)
        return t[n];
      if (n in _t)
        return _t[n](e);
    },
    has(t, n) {
      return n in t || n in _t;
    }
  })) : e.proxy;
}
const Tc = /(?:^|[-_])\w/g, $c = (e) => e.replace(Tc, (t) => t.toUpperCase()).replace(/[-_]/g, "");
function Gr(e, t = !0) {
  return M(e) ? e.displayName || e.name : e.name || t && e.__name;
}
function cn(e, t, n = !1) {
  let o = Gr(t);
  if (!o && t.__file) {
    const s = t.__file.match(/([^/\\]+)\.\w+$/);
    s && (o = s[1]);
  }
  if (!o && e) {
    const s = (r) => {
      for (const l in r)
        if (r[l] === t)
          return l;
    };
    o = s(e.components) || e.parent && s(
      e.parent.type.components
    ) || s(e.appContext.components);
  }
  return o ? $c(o) : n ? "App" : "Anonymous";
}
function qr(e) {
  return M(e) && "__vccOpts" in e;
}
const Wt = (e, t) => {
  const n = /* @__PURE__ */ Wi(e, t, Zt);
  if (process.env.NODE_ENV !== "production") {
    const o = Br();
    o && o.appContext.config.warnRecursiveComputed && (n._warnRecursive = !0);
  }
  return n;
};
function Ic(e, t, n) {
  try {
    Pn(-1);
    const o = arguments.length;
    return o === 2 ? U(t) && !T(t) ? Ct(t) ? xe(e, null, [t]) : xe(e, t) : xe(e, null, t) : (o > 3 ? n = Array.prototype.slice.call(arguments, 2) : o === 3 && Ct(n) && (n = [n]), xe(e, t, n));
  } finally {
    Pn(1);
  }
}
function Pc() {
  if (process.env.NODE_ENV === "production" || typeof window > "u")
    return;
  const e = { style: "color:#3ba776" }, t = { style: "color:#1677ff" }, n = { style: "color:#f5222d" }, o = { style: "color:#eb2f96" }, s = {
    __vue_custom_formatter: !0,
    header(p) {
      if (!U(p))
        return null;
      if (p.__isVue)
        return ["div", e, "VueInstance"];
      if (/* @__PURE__ */ ne(p)) {
        Ae();
        const g = p.value;
        return Me(), [
          "div",
          {},
          ["span", e, a(p)],
          "<",
          i(g),
          ">"
        ];
      } else {
        if (/* @__PURE__ */ ct(p))
          return [
            "div",
            {},
            ["span", e, /* @__PURE__ */ he(p) ? "ShallowReactive" : "Reactive"],
            "<",
            i(p),
            `>${/* @__PURE__ */ Re(p) ? " (readonly)" : ""}`
          ];
        if (/* @__PURE__ */ Re(p))
          return [
            "div",
            {},
            ["span", e, /* @__PURE__ */ he(p) ? "ShallowReadonly" : "Readonly"],
            "<",
            i(p),
            ">"
          ];
      }
      return null;
    },
    hasBody(p) {
      return p && p.__isVue;
    },
    body(p) {
      if (p && p.__isVue)
        return [
          "div",
          {},
          ...r(p.$)
        ];
    }
  };
  function r(p) {
    const g = [];
    p.type.props && p.props && g.push(l("props", /* @__PURE__ */ j(p.props))), p.setupState !== Y && g.push(l("setup", p.setupState)), p.data !== Y && g.push(l("data", /* @__PURE__ */ j(p.data)));
    const O = u(p, "computed");
    O && g.push(l("computed", O));
    const $ = u(p, "inject");
    return $ && g.push(l("injected", $)), g.push([
      "div",
      {},
      [
        "span",
        {
          style: o.style + ";opacity:0.66"
        },
        "$ (internal): "
      ],
      ["object", { object: p }]
    ]), g;
  }
  function l(p, g) {
    return g = Z({}, g), Object.keys(g).length ? [
      "div",
      { style: "line-height:1.25em;margin-bottom:0.6em" },
      [
        "div",
        {
          style: "color:#476582"
        },
        p
      ],
      [
        "div",
        {
          style: "padding-left:1.25em"
        },
        ...Object.keys(g).map((O) => [
          "div",
          {},
          ["span", o, O + ": "],
          i(g[O], !1)
        ])
      ]
    ] : ["span", {}];
  }
  function i(p, g = !0) {
    return typeof p == "number" ? ["span", t, p] : typeof p == "string" ? ["span", n, JSON.stringify(p)] : typeof p == "boolean" ? ["span", o, p] : U(p) ? ["object", { object: g ? /* @__PURE__ */ j(p) : p }] : ["span", n, String(p)];
  }
  function u(p, g) {
    const O = p.type;
    if (M(O))
      return;
    const $ = {};
    for (const D in p.ctx)
      d(O, D, g) && ($[D] = p.ctx[D]);
    return $;
  }
  function d(p, g, O) {
    const $ = p[O];
    if (T($) && $.includes(g) || U($) && g in $ || p.extends && d(p.extends, g, O) || p.mixins && p.mixins.some((D) => d(D, g, O)))
      return !0;
  }
  function a(p) {
    return /* @__PURE__ */ he(p) ? "ShallowRef" : p.effect ? "ComputedRef" : "Ref";
  }
  window.devtoolsFormatters ? window.devtoolsFormatters.push(s) : window.devtoolsFormatters = [s];
}
const ys = "3.5.29", et = process.env.NODE_ENV !== "production" ? b : ie;
process.env.NODE_ENV;
process.env.NODE_ENV;
/**
* @vue/runtime-dom v3.5.29
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/
let yo;
const Ns = typeof window < "u" && window.trustedTypes;
if (Ns)
  try {
    yo = /* @__PURE__ */ Ns.createPolicy("vue", {
      createHTML: (e) => e
    });
  } catch (e) {
    process.env.NODE_ENV !== "production" && et(`Error creating trusted types policy: ${e}`);
  }
const Yr = yo ? (e) => yo.createHTML(e) : (e) => e, Ac = "http://www.w3.org/2000/svg", Mc = "http://www.w3.org/1998/Math/MathML", Qe = typeof document < "u" ? document : null, bs = Qe && /* @__PURE__ */ Qe.createElement("template"), Rc = {
  insert: (e, t, n) => {
    t.insertBefore(e, n || null);
  },
  remove: (e) => {
    const t = e.parentNode;
    t && t.removeChild(e);
  },
  createElement: (e, t, n, o) => {
    const s = t === "svg" ? Qe.createElementNS(Ac, e) : t === "mathml" ? Qe.createElementNS(Mc, e) : n ? Qe.createElement(e, { is: n }) : Qe.createElement(e);
    return e === "select" && o && o.multiple != null && s.setAttribute("multiple", o.multiple), s;
  },
  createText: (e) => Qe.createTextNode(e),
  createComment: (e) => Qe.createComment(e),
  setText: (e, t) => {
    e.nodeValue = t;
  },
  setElementText: (e, t) => {
    e.textContent = t;
  },
  parentNode: (e) => e.parentNode,
  nextSibling: (e) => e.nextSibling,
  querySelector: (e) => Qe.querySelector(e),
  setScopeId(e, t) {
    e.setAttribute(t, "");
  },
  // __UNSAFE__
  // Reason: innerHTML.
  // Static content here can only come from compiled templates.
  // As long as the user only uses trusted templates, this is safe.
  insertStaticContent(e, t, n, o, s, r) {
    const l = n ? n.previousSibling : t.lastChild;
    if (s && (s === r || s.nextSibling))
      for (; t.insertBefore(s.cloneNode(!0), n), !(s === r || !(s = s.nextSibling)); )
        ;
    else {
      bs.innerHTML = Yr(
        o === "svg" ? `<svg>${e}</svg>` : o === "mathml" ? `<math>${e}</math>` : e
      );
      const i = bs.content;
      if (o === "svg" || o === "mathml") {
        const u = i.firstChild;
        for (; u.firstChild; )
          i.appendChild(u.firstChild);
        i.removeChild(u);
      }
      t.insertBefore(i, n);
    }
    return [
      // first
      l ? l.nextSibling : t.firstChild,
      // last
      n ? n.previousSibling : t.lastChild
    ];
  }
}, Fc = /* @__PURE__ */ Symbol("_vtc");
function jc(e, t, n) {
  const o = e[Fc];
  o && (t = (t ? [t, ...o] : [...o]).join(" ")), t == null ? e.removeAttribute("class") : n ? e.setAttribute("class", t) : e.className = t;
}
const Os = /* @__PURE__ */ Symbol("_vod"), Hc = /* @__PURE__ */ Symbol("_vsh"), kc = /* @__PURE__ */ Symbol(process.env.NODE_ENV !== "production" ? "CSS_VAR_TEXT" : ""), Lc = /(?:^|;)\s*display\s*:/;
function Wc(e, t, n) {
  const o = e.style, s = X(n);
  let r = !1;
  if (n && !s) {
    if (t)
      if (X(t))
        for (const l of t.split(";")) {
          const i = l.slice(0, l.indexOf(":")).trim();
          n[i] == null && Nn(o, i, "");
        }
      else
        for (const l in t)
          n[l] == null && Nn(o, l, "");
    for (const l in n)
      l === "display" && (r = !0), Nn(o, l, n[l]);
  } else if (s) {
    if (t !== n) {
      const l = o[kc];
      l && (n += ";" + l), o.cssText = n, r = Lc.test(n);
    }
  } else t && e.removeAttribute("style");
  Os in e && (e[Os] = r ? o.display : "", e[Hc] && (o.display = "none"));
}
const Bc = /[^\\];\s*$/, xs = /\s*!important$/;
function Nn(e, t, n) {
  if (T(n))
    n.forEach((o) => Nn(e, t, o));
  else if (n == null && (n = ""), process.env.NODE_ENV !== "production" && Bc.test(n) && et(
    `Unexpected semicolon at the end of '${t}' style value: '${n}'`
  ), t.startsWith("--"))
    e.setProperty(t, n);
  else {
    const o = Uc(e, t);
    xs.test(n) ? e.setProperty(
      ut(o),
      n.replace(xs, ""),
      "important"
    ) : e[o] = n;
  }
}
const ws = ["Webkit", "Moz", "ms"], oo = {};
function Uc(e, t) {
  const n = oo[t];
  if (n)
    return n;
  let o = $e(t);
  if (o !== "filter" && o in e)
    return oo[t] = o;
  o = Rn(o);
  for (let s = 0; s < ws.length; s++) {
    const r = ws[s] + o;
    if (r in e)
      return oo[t] = r;
  }
  return t;
}
const Ds = "http://www.w3.org/1999/xlink";
function Vs(e, t, n, o, s, r = hi(t)) {
  o && t.startsWith("xlink:") ? n == null ? e.removeAttributeNS(Ds, t.slice(6, t.length)) : e.setAttributeNS(Ds, t, n) : n == null || r && !js(n) ? e.removeAttribute(t) : e.setAttribute(
    t,
    r ? "" : qe(n) ? String(n) : n
  );
}
function Ss(e, t, n, o, s) {
  if (t === "innerHTML" || t === "textContent") {
    n != null && (e[t] = t === "innerHTML" ? Yr(n) : n);
    return;
  }
  const r = e.tagName;
  if (t === "value" && r !== "PROGRESS" && // custom elements may use _value internally
  !r.includes("-")) {
    const i = r === "OPTION" ? e.getAttribute("value") || "" : e.value, u = n == null ? (
      // #11647: value should be set as empty string for null and undefined,
      // but <input type="checkbox"> should be set as 'on'.
      e.type === "checkbox" ? "on" : ""
    ) : String(n);
    (i !== u || !("_value" in e)) && (e.value = u), n == null && e.removeAttribute(t), e._value = n;
    return;
  }
  let l = !1;
  if (n === "" || n == null) {
    const i = typeof e[t];
    i === "boolean" ? n = js(n) : n == null && i === "string" ? (n = "", l = !0) : i === "number" && (n = 0, l = !0);
  }
  try {
    e[t] = n;
  } catch (i) {
    process.env.NODE_ENV !== "production" && !l && et(
      `Failed setting prop "${t}" on <${r.toLowerCase()}>: value ${n} is invalid.`,
      i
    );
  }
  l && e.removeAttribute(s || t);
}
function Kc(e, t, n, o) {
  e.addEventListener(t, n, o);
}
function Gc(e, t, n, o) {
  e.removeEventListener(t, n, o);
}
const Cs = /* @__PURE__ */ Symbol("_vei");
function qc(e, t, n, o, s = null) {
  const r = e[Cs] || (e[Cs] = {}), l = r[t];
  if (o && l)
    l.value = process.env.NODE_ENV !== "production" ? $s(o, t) : o;
  else {
    const [i, u] = Yc(t);
    if (o) {
      const d = r[t] = Qc(
        process.env.NODE_ENV !== "production" ? $s(o, t) : o,
        s
      );
      Kc(e, i, d, u);
    } else l && (Gc(e, i, l, u), r[t] = void 0);
  }
}
const Ts = /(?:Once|Passive|Capture)$/;
function Yc(e) {
  let t;
  if (Ts.test(e)) {
    t = {};
    let o;
    for (; o = e.match(Ts); )
      e = e.slice(0, e.length - o[0].length), t[o[0].toLowerCase()] = !0;
  }
  return [e[2] === ":" ? e.slice(3) : ut(e.slice(2)), t];
}
let so = 0;
const Jc = /* @__PURE__ */ Promise.resolve(), zc = () => so || (Jc.then(() => so = 0), so = Date.now());
function Qc(e, t) {
  const n = (o) => {
    if (!o._vts)
      o._vts = Date.now();
    else if (o._vts <= n.attached)
      return;
    Ye(
      Xc(o, n.value),
      t,
      5,
      [o]
    );
  };
  return n.value = e, n.attached = zc(), n;
}
function $s(e, t) {
  return M(e) || T(e) ? e : (et(
    `Wrong type passed as event handler to ${t} - did you forget @ or : in front of your prop?
Expected function or array of functions, received type ${typeof e}.`
  ), ie);
}
function Xc(e, t) {
  if (T(t)) {
    const n = e.stopImmediatePropagation;
    return e.stopImmediatePropagation = () => {
      n.call(e), e._stopped = !0;
    }, t.map(
      (o) => (s) => !s._stopped && o && o(s)
    );
  } else
    return t;
}
const Is = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && // lowercase letter
e.charCodeAt(2) > 96 && e.charCodeAt(2) < 123, Zc = (e, t, n, o, s, r) => {
  const l = s === "svg";
  t === "class" ? jc(e, o, l) : t === "style" ? Wc(e, n, o) : en(t) ? bn(t) || qc(e, t, n, o, r) : (t[0] === "." ? (t = t.slice(1), !0) : t[0] === "^" ? (t = t.slice(1), !1) : eu(e, t, o, l)) ? (Ss(e, t, o), !e.tagName.includes("-") && (t === "value" || t === "checked" || t === "selected") && Vs(e, t, o, l, r, t !== "value")) : /* #11081 force set props for possible async custom element */ e._isVueCE && (/[A-Z]/.test(t) || !X(o)) ? Ss(e, $e(t), o, r, t) : (t === "true-value" ? e._trueValue = o : t === "false-value" && (e._falseValue = o), Vs(e, t, o, l));
};
function eu(e, t, n, o) {
  if (o)
    return !!(t === "innerHTML" || t === "textContent" || t in e && Is(t) && M(n));
  if (t === "spellcheck" || t === "draggable" || t === "translate" || t === "autocorrect" || t === "sandbox" && e.tagName === "IFRAME" || t === "form" || t === "list" && e.tagName === "INPUT" || t === "type" && e.tagName === "TEXTAREA")
    return !1;
  if (t === "width" || t === "height") {
    const s = e.tagName;
    if (s === "IMG" || s === "VIDEO" || s === "CANVAS" || s === "SOURCE")
      return !1;
  }
  return Is(t) && X(n) ? !1 : t in e;
}
const tu = /* @__PURE__ */ Z({ patchProp: Zc }, Rc);
let Ps;
function nu() {
  return Ps || (Ps = fc(tu));
}
const ou = ((...e) => {
  const t = nu().createApp(...e);
  process.env.NODE_ENV !== "production" && (ru(t), iu(t));
  const { mount: n } = t;
  return t.mount = (o) => {
    const s = lu(o);
    if (!s) return;
    const r = t._component;
    !M(r) && !r.render && !r.template && (r.template = s.innerHTML), s.nodeType === 1 && (s.textContent = "");
    const l = n(s, !1, su(s));
    return s instanceof Element && (s.removeAttribute("v-cloak"), s.setAttribute("data-v-app", "")), l;
  }, t;
});
function su(e) {
  if (e instanceof SVGElement)
    return "svg";
  if (typeof MathMLElement == "function" && e instanceof MathMLElement)
    return "mathml";
}
function ru(e) {
  Object.defineProperty(e.config, "isNativeTag", {
    value: (t) => fi(t) || ai(t) || pi(t),
    writable: !1
  });
}
function iu(e) {
  {
    const t = e.config.isCustomElement;
    Object.defineProperty(e.config, "isCustomElement", {
      get() {
        return t;
      },
      set() {
        et(
          "The `isCustomElement` config option is deprecated. Use `compilerOptions.isCustomElement` instead."
        );
      }
    });
    const n = e.config.compilerOptions, o = 'The `compilerOptions` config option is only respected when using a build of Vue.js that includes the runtime compiler (aka "full build"). Since you are using the runtime-only build, `compilerOptions` must be passed to `@vue/compiler-dom` in the build setup instead.\n- For vue-loader: pass it via vue-loader\'s `compilerOptions` loader option.\n- For vue-cli: see https://cli.vuejs.org/guide/webpack.html#modifying-options-of-a-loader\n- For vite: pass it via @vitejs/plugin-vue options. See https://github.com/vitejs/vite-plugin-vue/tree/main/packages/plugin-vue#example-for-passing-options-to-vuecompiler-sfc';
    Object.defineProperty(e.config, "compilerOptions", {
      get() {
        return et(o), n;
      },
      set() {
        et(o);
      }
    });
  }
}
function lu(e) {
  if (X(e)) {
    const t = document.querySelector(e);
    return process.env.NODE_ENV !== "production" && !t && et(
      `Failed to mount app: mount target selector "${e}" returned null.`
    ), t;
  }
  return process.env.NODE_ENV !== "production" && window.ShadowRoot && e instanceof window.ShadowRoot && e.mode === "closed" && et(
    'mounting on a ShadowRoot with `{mode: "closed"}` may lead to unpredictable bugs'
  ), e;
}
/**
* vue v3.5.29
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/
function cu() {
  Pc();
}
process.env.NODE_ENV !== "production" && cu();
const uu = { class: "ease-curve-root" }, fu = ["x1", "x2", "y2"], au = ["y1", "x2", "y2"], pu = ["x1", "y1", "x2", "y2"], du = ["d"], hu = ["x1", "y1", "x2", "y2"], gu = ["x1", "y1", "x2", "y2"], vu = ["cx", "cy"], mu = ["cx", "cy"], _u = ["cx", "cy"], Eu = ["cx", "cy"], yu = {
  x: "100",
  y: "14",
  "text-anchor": "middle",
  fill: "#889",
  "font-size": "10",
  "font-family": "sans-serif"
}, Nu = {
  key: 0,
  class: "preset-grid"
}, bu = ["onClick", "title"], Ou = {
  viewBox: "0 0 30 30",
  class: "thumb-svg"
}, xu = ["d"], wu = { class: "thumb-label" }, _e = 20, Ve = 160, Du = /* @__PURE__ */ El({
  __name: "EaseCurveWidget",
  props: {
    node: {},
    presetWidget: {}
  },
  setup(e) {
    const t = e, n = {
      linear: [0, 0, 1, 1],
      easeInQuad: [0.55, 0.085, 0.68, 0.53],
      easeOutQuad: [0.25, 0.46, 0.45, 0.94],
      easeInOutQuad: [0.455, 0.03, 0.515, 0.955],
      easeInCubic: [0.55, 0.055, 0.675, 0.19],
      easeOutCubic: [0.215, 0.61, 0.355, 1],
      easeInOutCubic: [0.645, 0.045, 0.355, 1],
      easeInQuart: [0.895, 0.03, 0.685, 0.22],
      easeOutQuart: [0.165, 0.84, 0.44, 1],
      easeInOutQuart: [0.77, 0, 0.175, 1],
      easeInQuint: [0.755, 0.05, 0.855, 0.06],
      easeOutQuint: [0.23, 1, 0.32, 1],
      easeInOutQuint: [0.86, 0, 0.07, 1],
      easeInSine: [0.47, 0, 0.745, 0.715],
      easeOutSine: [0.39, 0.575, 0.565, 1],
      easeInOutSine: [0.445, 0.05, 0.55, 0.95],
      easeInExpo: [0.95, 0.05, 0.795, 0.035],
      easeOutExpo: [0.19, 1, 0.22, 1],
      easeInOutExpo: [1, 0, 0, 1],
      easeInCirc: [0.6, 0.04, 0.98, 0.335],
      easeOutCirc: [0.075, 0.82, 0.165, 1],
      easeInOutCirc: [0.785, 0.135, 0.15, 0.86],
      easeInBack: [0.6, -0.28, 0.735, 0.045],
      easeOutBack: [0.175, 0.885, 0.32, 1.275],
      easeInOutBack: [0.68, -0.55, 0.265, 1.55],
      easeInElastic: [0.5, -0.5, 0.75, -0.5],
      easeOutElastic: [0.25, 1.5, 0.5, 1.5],
      easeInOutElastic: [0.5, -0.5, 0.5, 1.5],
      easeInBounce: [0.5, -0.3, 0.7, -0.3],
      easeOutBounce: [0.3, 1.3, 0.5, 1.3],
      easeInOutBounce: [0.5, -0.3, 0.5, 1.3]
    }, o = Object.keys(n), s = /* @__PURE__ */ Ft(t.presetWidget.value || "linear"), r = /* @__PURE__ */ Ft(!1), l = /* @__PURE__ */ Ft(null), i = /* @__PURE__ */ Ft({ x1: 0.42, y1: 0, x2: 0.58, y2: 1 }), u = Wt(() => s.value === "custom");
    function d(V) {
      return _e + V * Ve;
    }
    function a(V) {
      return _e + (1 - V) * Ve;
    }
    function p(V, I, P, k) {
      return `M ${d(0)} ${a(0)} C ${d(V)} ${a(I)}, ${d(P)} ${a(k)}, ${d(1)} ${a(1)}`;
    }
    const g = Wt(() => {
      if (u.value)
        return p(i.value.x1, i.value.y1, i.value.x2, i.value.y2);
      const V = n[s.value] || [0, 0, 1, 1];
      return p(V[0], V[1], V[2], V[3]);
    }), O = Wt(() => ({ x: d(i.value.x1), y: a(i.value.y1) })), $ = Wt(() => ({ x: d(i.value.x2), y: a(i.value.y2) }));
    function D(V) {
      var I;
      return (I = t.node.widgets) == null ? void 0 : I.find((P) => P.name === V);
    }
    function Q() {
      s.value = t.presetWidget.value || "linear";
      const V = D("x1"), I = D("y1"), P = D("x2"), k = D("y2");
      V && (i.value.x1 = V.value), I && (i.value.y1 = I.value), P && (i.value.x2 = P.value), k && (i.value.y2 = k.value);
    }
    function J() {
      const V = D("x1"), I = D("y1"), P = D("x2"), k = D("y2");
      V && (V.value = i.value.x1), I && (I.value = i.value.y1), P && (P.value = i.value.x2), k && (k.value = i.value.y2);
    }
    function K(V) {
      var I, P;
      if (t.presetWidget.value = V, s.value = V, V !== "custom") {
        const k = n[V];
        k && (i.value = { x1: k[0], y1: k[1], x2: k[2], y2: k[3] }, J());
      }
      (P = (I = t.node).setDirtyCanvas) == null || P.call(I, !0, !0);
    }
    const L = /* @__PURE__ */ Ft(null);
    function ue(V) {
      const I = L.value;
      if (!I) return { x: 0, y: 0 };
      const P = I.getBoundingClientRect(), k = 200 / P.width, oe = 200 / P.height;
      return {
        x: (V.clientX - P.left) * k,
        y: (V.clientY - P.top) * oe
      };
    }
    function C(V) {
      return Math.max(0, Math.min(1, (V - _e) / Ve));
    }
    function ee(V) {
      return Math.max(-1, Math.min(2, 1 - (V - _e) / Ve));
    }
    function fe(V, I) {
      var P, k;
      u.value && (I.preventDefault(), I.stopPropagation(), l.value = V, (k = (P = I.target) == null ? void 0 : P.setPointerCapture) == null || k.call(P, I.pointerId));
    }
    function le(V) {
      if (!l.value) return;
      V.preventDefault(), V.stopPropagation();
      const I = ue(V), P = C(I.x), k = ee(I.y);
      l.value === "p1" ? (i.value.x1 = Math.round(P * 100) / 100, i.value.y1 = Math.round(k * 100) / 100) : (i.value.x2 = Math.round(P * 100) / 100, i.value.y2 = Math.round(k * 100) / 100), J();
    }
    function ce() {
      var V, I;
      l.value = null, (I = (V = t.node).setDirtyCanvas) == null || I.call(V, !0, !0);
    }
    function Se(V) {
      const I = n[V] || [0, 0, 1, 1], P = 30, k = 3, oe = P - 2 * k, H = (ge) => k + ge * oe, F = (ge) => k + (1 - ge) * oe;
      return `M ${H(0)} ${F(0)} C ${H(I[0])} ${F(I[1])}, ${H(I[2])} ${F(I[3])}, ${H(1)} ${F(1)}`;
    }
    let De;
    return yr(() => {
      Q(), De = window.setInterval(Q, 300);
    }), jo(() => {
      De !== void 0 && window.clearInterval(De);
    }), (V, I) => (We(), ze("div", uu, [
      (We(), ze("svg", {
        ref_key: "svgRef",
        ref: L,
        viewBox: "0 0 200 200",
        class: "curve-svg",
        onPointermove: le,
        onPointerup: ce,
        onPointerleave: ce
      }, [
        I[3] || (I[3] = te("rect", {
          x: "0",
          y: "0",
          width: "200",
          height: "200",
          fill: "#1a1a2e",
          rx: "6"
        }, null, -1)),
        (We(), ze(Ne, null, eo(3, (P) => te("line", {
          key: "gv" + P,
          x1: _e + P * Ve / 4,
          y1: _e,
          x2: _e + P * Ve / 4,
          y2: _e + Ve,
          stroke: "#2a2a4a",
          "stroke-width": "0.5"
        }, null, 8, fu)), 64)),
        (We(), ze(Ne, null, eo(3, (P) => te("line", {
          key: "gh" + P,
          x1: _e,
          y1: _e + P * Ve / 4,
          x2: _e + Ve,
          y2: _e + P * Ve / 4,
          stroke: "#2a2a4a",
          "stroke-width": "0.5"
        }, null, 8, au)), 64)),
        te("rect", {
          x: _e,
          y: _e,
          width: Ve,
          height: Ve,
          fill: "none",
          stroke: "#3a3a5a",
          "stroke-width": "1"
        }),
        te("line", {
          x1: d(0),
          y1: a(0),
          x2: d(1),
          y2: a(1),
          stroke: "#3a3a5a",
          "stroke-width": "1",
          "stroke-dasharray": "4 4"
        }, null, 8, pu),
        te("path", {
          d: g.value,
          fill: "none",
          stroke: "#6cf",
          "stroke-width": "2.5",
          "stroke-linecap": "round"
        }, null, 8, du),
        u.value ? (We(), ze(Ne, { key: 0 }, [
          te("line", {
            x1: d(0),
            y1: a(0),
            x2: O.value.x,
            y2: O.value.y,
            stroke: "#f80",
            "stroke-width": "1",
            "stroke-dasharray": "3 3",
            opacity: "0.7"
          }, null, 8, hu),
          te("line", {
            x1: d(1),
            y1: a(1),
            x2: $.value.x,
            y2: $.value.y,
            stroke: "#f80",
            "stroke-width": "1",
            "stroke-dasharray": "3 3",
            opacity: "0.7"
          }, null, 8, gu),
          te("circle", {
            cx: O.value.x,
            cy: O.value.y,
            r: "6",
            fill: "#f80",
            stroke: "#fff",
            "stroke-width": "1.5",
            class: "handle",
            onPointerdown: I[0] || (I[0] = (P) => fe("p1", P))
          }, null, 40, vu),
          te("circle", {
            cx: $.value.x,
            cy: $.value.y,
            r: "6",
            fill: "#f80",
            stroke: "#fff",
            "stroke-width": "1.5",
            class: "handle",
            onPointerdown: I[1] || (I[1] = (P) => fe("p2", P))
          }, null, 40, mu)
        ], 64)) : vs("", !0),
        te("circle", {
          cx: d(0),
          cy: a(0),
          r: "3",
          fill: "#6cf"
        }, null, 8, _u),
        te("circle", {
          cx: d(1),
          cy: a(1),
          r: "3",
          fill: "#6cf"
        }, null, 8, Eu),
        te("text", yu, dn(u.value ? `custom (${i.value.x1}, ${i.value.y1}, ${i.value.x2}, ${i.value.y2})` : s.value), 1)
      ], 544)),
      te("button", {
        class: "grid-toggle",
        onClick: I[2] || (I[2] = (P) => r.value = !r.value)
      }, dn(r.value ? "Hide presets" : "Show presets"), 1),
      r.value ? (We(), ze("div", Nu, [
        (We(!0), ze(Ne, null, eo(or(o), (P) => (We(), ze("div", {
          key: P,
          class: Fn(["preset-thumb", { active: s.value === P }]),
          onClick: (k) => K(P),
          title: P
        }, [
          (We(), ze("svg", Ou, [
            I[4] || (I[4] = te("rect", {
              x: "0",
              y: "0",
              width: "30",
              height: "30",
              fill: "none"
            }, null, -1)),
            I[5] || (I[5] = te("line", {
              x1: "3",
              y1: "27",
              x2: "27",
              y2: "3",
              stroke: "#2a2a4a",
              "stroke-width": "0.5",
              "stroke-dasharray": "2 2"
            }, null, -1)),
            te("path", {
              d: Se(P),
              fill: "none",
              stroke: "#6cf",
              "stroke-width": "1.5",
              "stroke-linecap": "round"
            }, null, 8, xu)
          ])),
          te("span", wu, dn(P.replace(/ease/i, "").replace(/In|Out/g, (k) => k + " ").trim() || "linear"), 1)
        ], 10, bu))), 128))
      ])) : vs("", !0)
    ]));
  }
}), Vu = (e, t) => {
  const n = e.__vccOpts || e;
  for (const [o, s] of t)
    n[o] = s;
  return n;
}, Su = /* @__PURE__ */ Vu(Du, [["__scopeId", "data-v-a2535006"]]), Cu = ["EaseCurve", "ApplyEasingToFloats"];
Qr.registerExtension({
  name: "nodesweet.easeCurve",
  nodeCreated(e) {
    var r;
    if (!Cu.includes(e.comfyClass)) return;
    const t = (r = e.widgets) == null ? void 0 : r.find((l) => l.name === "preset");
    if (!t) return;
    const n = document.createElement("div");
    n.style.width = "100%", ou({
      render() {
        return Ic(Su, {
          node: e,
          presetWidget: t
        });
      }
    }).mount(n);
    const s = e.addDOMWidget("curve_preview", "div", n, {
      serialize: !1,
      hideOnZoom: !1
    });
    s.computeSize = () => [200, 260], e.setSize([320, e.computeSize()[1]]);
  }
});
