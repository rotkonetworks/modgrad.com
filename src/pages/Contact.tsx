import { createSignal, Show } from "solid-js";
import { useDocMeta } from "@/lib/meta";
import { reveal } from "@/lib/reveal";
import { site } from "@/data/site";

false && reveal;

type Status = "idle" | "sending" | "sent" | "error";

const emailOk = (e: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e);

export default function Contact() {
  useDocMeta(() => ({
    title: "Community & contact",
    description:
      "Ask questions in the open on GitHub Discussions, file issues, or get in touch about building on modgrad or funding where it's headed.",
    path: "/contact",
  }));

  const [name, setName] = createSignal("");
  const [email, setEmail] = createSignal("");
  const [message, setMessage] = createSignal("");
  const [company, setCompany] = createSignal(""); // honeypot
  const [status, setStatus] = createSignal<Status>("idle");
  const [error, setError] = createSignal("");

  const submit = async (e: Event) => {
    e.preventDefault();
    if (!emailOk(email())) return fail("Please enter a valid email so we can reply.");
    if (message().trim().length < 3) return fail("Add a short message.");
    setStatus("sending");
    setError("");
    try {
      const r = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name().trim(),
          email: email().trim(),
          message: message().trim(),
          company: company(),
        }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.error || `Something went wrong (${r.status}).`);
      }
      setStatus("sent");
    } catch (err) {
      fail(err instanceof Error ? err.message : "Network error. Please try again.");
    }
  };

  function fail(msg: string) {
    setStatus("error");
    setError(msg);
  }

  return (
    <div class="container-page py-16 max-w-[920px]">
      <div class="eyebrow mb-3">Community</div>
      <h1 class="text-[clamp(2rem,5vw,3rem)] tracking-[-0.03em] leading-[1.08]">
        Get in <span class="grad-text">touch.</span>
      </h1>
      <p class="mt-4 text-dim text-[1.05rem] max-w-[58ch] leading-relaxed">
        Technical questions are best asked in the open, where the answers help
        everyone. For private inquiries — building on modgrad, or funding where it's
        headed — use the form.
      </p>

      <div use:reveal class="grid lg:grid-cols-[1.1fr_0.9fr] gap-5 mt-10">
        {/* form */}
        <div class="card glow-card">
          <Show
            when={status() !== "sent"}
            fallback={
              <div class="py-8 text-center">
                <div class="text-3xl grad-text font-mono mb-3">∇</div>
                <h2 class="text-xl mb-2">Message sent.</h2>
                <p class="text-dim text-sm leading-relaxed max-w-[34ch] mx-auto">
                  Thanks — it's on its way to the team and we'll reply to the address
                  you gave.
                </p>
              </div>
            }
          >
            <form onSubmit={submit} class="flex flex-col gap-4">
              <div>
                <label class="lbl" for="c-name">Name <span class="text-mute">(optional)</span></label>
                <input id="c-name" class="field" value={name()} onInput={(e) => setName(e.currentTarget.value)} placeholder="Ada Lovelace" autocomplete="name" />
              </div>
              <div>
                <label class="lbl" for="c-email">Email</label>
                <input id="c-email" class="field" type="email" required value={email()} onInput={(e) => setEmail(e.currentTarget.value)} placeholder="you@company.com" autocomplete="email" />
              </div>
              <div>
                <label class="lbl" for="c-msg">Message</label>
                <textarea id="c-msg" class="field" required value={message()} onInput={(e) => setMessage(e.currentTarget.value)} placeholder="What are you working on?" />
              </div>

              {/* honeypot — hidden from humans */}
              <input class="hp" tabindex="-1" autocomplete="off" aria-hidden="true" value={company()} onInput={(e) => setCompany(e.currentTarget.value)} name="company" />

              <Show when={status() === "error"}>
                <div class="text-sm text-warn">{error()}</div>
              </Show>

              <button type="submit" class="btn btn-primary w-full justify-center" disabled={status() === "sending"}>
                {status() === "sending" ? "Sending…" : "Send message →"}
              </button>
              <p class="text-xs text-mute leading-relaxed">
                Goes straight to the modgrad inbox. We never share your address.
              </p>
            </form>
          </Show>
        </div>

        {/* open forum */}
        <div class="flex flex-col gap-4">
          <div class="card card-hover">
            <div class="eyebrow mb-3">Open forum</div>
            <h2 class="text-lg mb-2">Ask in public</h2>
            <p class="text-sm text-dim leading-relaxed mb-4">
              Questions, ideas, and design discussion happen on GitHub Discussions —
              searchable, and open to everyone.
            </p>
            <a href={site.discussions} target="_blank" rel="noopener" class="btn btn-ghost w-full justify-center">
              Open Discussions ↗
            </a>
          </div>
          <div class="card card-hover">
            <div class="eyebrow mb-3">Found a bug?</div>
            <h2 class="text-lg mb-2">File an issue</h2>
            <p class="text-sm text-dim leading-relaxed mb-4">
              Bug reports and feature requests live on the issue tracker, next to the
              source.
            </p>
            <div class="flex gap-2">
              <a href={site.issues} target="_blank" rel="noopener" class="btn btn-ghost flex-1 justify-center">Issues ↗</a>
              <a href={site.repo} target="_blank" rel="noopener" class="btn btn-ghost flex-1 justify-center">Source ↗</a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
