import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "antd/dist/reset.css";

/**
 * 这是 React 应用入口
 *
 * ReactDOM.createRoot(...).render(...)
 * 的作用：
 * - 把 <App /> 这个 React 页面挂到浏览器里的 root 节点上
 */

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);