"use client";

import { useEffect } from "react";

export default function Analytics() {
  useEffect(() => {
    const ga = process.env.NEXT_PUBLIC_GA4_ID;
    if (ga) {
      const s1 = document.createElement("script");
      s1.async = true;
      s1.src = `https://www.googletagmanager.com/gtag/js?id=${ga}`;
      document.head.appendChild(s1);

      const s2 = document.createElement("script");
      s2.innerHTML = `window.dataLayer = window.dataLayer || []; function gtag(){dataLayer.push(arguments);} gtag('js', new Date()); gtag('config', '${ga}');`;
      document.head.appendChild(s2);
    }

    const pixel = process.env.NEXT_PUBLIC_META_PIXEL_ID;
    if (pixel) {
      const s = document.createElement("script");
      s.innerHTML = `!function(f,b,e,v,n,t,s) {if(f.fbq)return;n=f.fbq=function(){n.callMethod? n.callMethod.apply(n,arguments):n.queue.push(arguments)}; if(!f._fbq)f._fbq=n; n.push=n; n.loaded=!0;n.version='2.0'; n.queue=[]; t=b.createElement(e); t.async=!0; t.src=v; s=b.getElementsByTagName(e)[0]; s.parentNode.insertBefore(t,s)}(window, document,'script','https://connect.facebook.net/en_US/fbevents.js'); fbq('init','${pixel}'); fbq('track','PageView');`;
      document.head.appendChild(s);
    }
  }, []);

  return null;
}