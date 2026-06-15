import { Router, Route, useLocation } from "@solidjs/router";
import { createEffect, lazy, Suspense, type JSX } from "solid-js";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const Home = lazy(() => import("@/pages/Home"));
const Docs = lazy(() => import("@/pages/Docs"));
const Contact = lazy(() => import("@/pages/Contact"));

function Layout(props: { children?: JSX.Element }) {
  const loc = useLocation();
  // Scroll to top on navigation, unless linking to an in-page anchor.
  createEffect((prev) => {
    const path = loc.pathname;
    if (prev && prev !== path && !loc.hash) window.scrollTo({ top: 0 });
    return path;
  });

  return (
    <>
      <Header />
      <main>
        <Suspense>{props.children}</Suspense>
      </main>
      <Footer />
    </>
  );
}

export default function App() {
  return (
    <Router root={Layout}>
      <Route path="/" component={Home} />
      <Route path="/docs" component={Docs} />
      <Route path="/docs/:slug" component={Docs} />
      <Route path="/contact" component={Contact} />
      <Route path="*" component={Docs} />
    </Router>
  );
}
