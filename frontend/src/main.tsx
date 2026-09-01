import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/literata";
import "@fontsource-variable/instrument-sans";
import "@fontsource-variable/spline-sans-mono";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
