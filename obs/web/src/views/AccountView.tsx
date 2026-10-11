import { createMemo, createSignal, For, Show } from 'solid-js';
import { api, projectNameProblem, type Cred, type MintedToken } from '@/logic/account';
import { webauthn } from '@/logic/account';
import { me, meError, refreshMe, signOut, goLogin, heldAnonTokens } from '@/stores/account';
import { showToast } from '@/stores/ui';
import { EmailMethod, PasskeyMethod, SshMethod, ZafuMethod } from '@/components/SignInMethods';
import { ConfirmButton, CopyLine, Msg, When, useAction } from '@/components/PortalBits';

const KIND: Record<string, string> = { ssh: 'SSH key', email: 'email', zafu: 'zafu', passkey: 'passkey' };
const boardHref = (project: string) => location.pathname + location.search + '#project=' + encodeURIComponent(project);

/** #view=account: credentials, projects, API tokens, sessions, account deletion */
export default function AccountView() {
  return (
    <section class="portal">
      <Show when={me()} fallback={
        <>
          <h2>Account</h2>
          <Show when={meError()} fallback={<p class="note">loading…</p>}>
            <div class="err" role="alert">{meError()}</div>
            <div class="prow"><button type="button" title="load the account again" aria-label="retry loading the account" onClick={() => void refreshMe()}>retry</button>
              <button type="button" title="go to the sign-in page" aria-label="go to sign in" onClick={goLogin}>sign in</button></div>
          </Show>
        </>
      }>{(m) => (
        <>
          <h2>Account</h2>
          <p class="note"><span class="mono acct">{m().account.id}</span> created <When unix={m().account.created} /></p>
          <Credentials />
          <Projects />
          <Tokens />
          <Sessions />
          <DeleteAccount />
        </>
      )}</Show>
    </section>
  );
}

function Credentials() {
  const a = useAction();
  const remove = (c: Cred) => a.run(async () => { await api.removeCred(c.id); await refreshMe(); showToast(`removed ${KIND[c.kind] || c.kind}`); });
  return (
    <>
      <h2>Credentials</h2>
      <p class="note">Any of these signs in to this account. Keep at least two, so losing one is not losing the account.</p>
      <div class="ptwrap"><table class="ptable">
        <thead><tr><th>kind</th><th>label</th><th>added</th><th>last used</th><th /></tr></thead>
        <tbody><For each={me()!.creds}>{(c) => (
          <tr>
            <td>{KIND[c.kind] || c.kind}</td>
            <td class="lab mono" title={c.label}>{c.label || '-'}</td>
            <td data-h="added"><When unix={c.added} /></td>
            <td data-h="last used"><When unix={c.last_used} /></td>
            <td class="act"><ConfirmButton label="remove" confirm="remove it" what={`${KIND[c.kind] || c.kind} ${c.label}`} run={() => remove(c)} /></td>
          </tr>
        )}</For></tbody>
      </table></div>
      <Msg err={a.err()} />
      <h3 class="psub">Add another</h3>
      <div class="pgrid">
        <div class="pbox"><h3>Email</h3><EmailMethod /></div>
        <Show when={webauthn().available()}><div class="pbox"><h3>Passkey</h3><PasskeyMethod mode="add" /></div></Show>
        <div class="pbox"><h3>zafu wallet</h3><ZafuMethod mode="add" /></div>
        <div class="pbox"><h3>SSH key</h3><SshAdd /><SshMethod /></div>
      </div>
    </>
  );
}

function SshAdd() {
  const [key, setKey] = createSignal('');
  const a = useAction();
  const add = (e: Event) => {
    e.preventDefault();
    const k = key().trim();
    if (!/^(ssh-(ed25519|rsa|dss)|ecdsa-sha2-nistp\d+|sk-(ssh-ed25519|ecdsa-sha2-nistp256)@openssh\.com) [A-Za-z0-9+/=]+/.test(k)) { a.setErr('Paste one public key line, e.g. the contents of ~/.ssh/id_ed25519.pub.'); return; }
    void a.run(async () => { await api.addSshKey(k); setKey(''); await refreshMe(); showToast('SSH key added'); });
  };
  return (
    <form onSubmit={add}>
      <textarea rows={3} placeholder="ssh-ed25519 AAAA… you@host" name="ssh-key" aria-label="SSH public key" spellcheck={false} value={key()} onInput={(e) => setKey(e.currentTarget.value)} />
      <div class="prow"><button type="submit" class="pri" disabled={a.busy() || !key().trim()} title="add this public key to the account" aria-label="add SSH key">{a.busy() ? 'adding…' : 'Add key'}</button>
        <span class="note">or run, with that key loaded:</span></div>
      <Msg err={a.err()} />
    </form>
  );
}

function Projects() {
  const [name, setName] = createSignal('');
  const [vis, setVis] = createSignal<'private' | 'public'>('private');
  const create = useAction(), row = useAction(), claim = useAction();
  const problem = () => (name() ? projectNameProblem(name()) : null);
  const owned = createMemo(() => new Set(me()!.projects.filter((p) => !p.anon).map((p) => p.name)));
  const claimable = createMemo(() => (me(), heldAnonTokens().filter((t) => t.project.startsWith('anon-') && !owned().has(t.project))));
  const submit = (e: Event) => {
    e.preventDefault();
    const n = name().trim(), p = projectNameProblem(n);
    if (p) { create.setErr(p); return; }
    void create.run(async () => { await api.createProject(n, vis()); setName(''); await refreshMe(); showToast(`project ${n} created`); });
  };
  return (
    <>
      <h2>Projects</h2>
      <Show when={me()!.projects.length} fallback={<p class="note">No projects yet.</p>}>
        <div class="ptwrap"><table class="ptable">
          <thead><tr><th>name</th><th>visibility</th><th>runs</th><th>last run</th><th /></tr></thead>
          <tbody><For each={me()!.projects}>{(p) => (
            <tr>
              <td class="mono"><a href={boardHref(p.name)} title={'open the board of ' + p.name}>{p.name}</a>
                <Show when={p.anon}> <span class="tag" title="anonymous project: claim it below to keep it">anon{p.expires ? ', expires ' : ''}<Show when={p.expires}><When unix={p.expires} /></Show></span></Show></td>
              <td data-h="visibility">{p.visibility}</td>
              <td class="num" data-h="runs">{p.runs}</td>
              <td data-h="last run"><When unix={p.last_unix} never="-" /></td>
              <td class="act">
                <button type="button" disabled={row.busy()} title={p.visibility === 'private' ? 'anyone with the URL can read it' : 'only you and project tokens can read it'}
                  aria-label={(p.visibility === 'private' ? 'make public: ' : 'make private: ') + p.name}
                  onClick={() => row.run(async () => { await api.setVisibility(p.name, p.visibility === 'private' ? 'public' : 'private'); await refreshMe(); })}>
                  {p.visibility === 'private' ? 'make public' : 'make private'}</button>
                <a class="btnlink" href={boardHref(p.name)} title={'open the board of ' + p.name} aria-label={'open board: ' + p.name}>open board</a>
              </td>
            </tr>
          )}</For></tbody>
        </table></div>
      </Show>
      <Msg err={row.err()} />
      <form class="prow" onSubmit={submit}>
        <input type="text" class="mono" placeholder="new-project-name" name="project-name" aria-label="new project name" spellcheck={false} autocomplete="off" maxLength={40} value={name()} onInput={(e) => { setName(e.currentTarget.value.toLowerCase()); create.setErr(''); }} />
        <label>visibility <select name="project-visibility" aria-label="new project visibility" value={vis()} onChange={(e) => setVis(e.currentTarget.value as 'private' | 'public')}>
          <option value="private">private</option><option value="public">public</option></select></label>
        <button type="submit" class="pri" disabled={create.busy() || !name() || !!problem()} title="create the project" aria-label="create project">{create.busy() ? 'creating…' : 'Create project'}</button>
      </form>
      <Msg err={create.err() || problem() || ''} />
      <p class="note">Private: readable with your session or a project token. Public: anyone with the URL.</p>
      <Show when={claimable().length}>
        <h3 class="psub">Anonymous boards in this browser</h3>
        <p class="note">This tab holds the write token of these anonymous projects. Claiming one makes it yours: it keeps its name and stops expiring.</p>
        <For each={claimable()}>{(t) => (
          <div class="prow">
            <a class="mono" href={boardHref(t.project)} title={'open the board of ' + t.project}>{t.project}</a>
            <button type="button" class="pri" disabled={claim.busy()} title={'make ' + t.project + ' a project of this account'} aria-label={'claim ' + t.project}
              onClick={() => claim.run(async () => { await api.claim(t.project, t.token); await refreshMe(); showToast(`${t.project} is now yours`); })}>claim</button>
          </div>
        )}</For>
        <Msg err={claim.err()} />
      </Show>
    </>
  );
}

function Tokens() {
  const projects = () => me()!.projects.map((p) => p.name);
  const [project, setProject] = createSignal('');
  const [label, setLabel] = createSignal('');
  const [days, setDays] = createSignal('90');
  const [minted, setMinted] = createSignal<(MintedToken & { project: string }) | null>(null);
  const mint = useAction(), revoke = useAction();
  const target = () => project() || projects()[0] || '';
  const submit = (e: Event) => {
    e.preventDefault();
    const p = target();
    if (!p) { mint.setErr('Create a project first.'); return; }
    void mint.run(async () => {
      const r = await api.mintToken(p, label().trim(), days() ? +days() : undefined);
      setMinted({ ...r, project: p }); setLabel(''); await refreshMe();
    });
  };
  return (
    <>
      <h2>API tokens</h2>
      <p class="note">A project token writes runs to one project and reads it when it is private. Use it from training machines and CI.</p>
      <Show when={minted()}>{(t) => (
        <div class="psecret" role="status">
          <b>New token for {t().project}.</b> It is shown once; copy it now.
          <CopyLine text={t().token} what="the token" />
          <div class="note">run with it:</div>
          <CopyLine text={`MODGRAD_TOKEN=${t().token} MODGRAD_PROJECT=${t().project} modgrad run -- python train.py`} what="the run command" />
          <div class="prow"><button type="button" title="hide the token; it cannot be shown again" aria-label="done, hide the token" onClick={() => setMinted(null)}>done, hide it</button></div>
        </div>
      )}</Show>
      <Show when={me()!.tokens.length} fallback={<p class="note">No tokens.</p>}>
        <div class="ptwrap"><table class="ptable">
          <thead><tr><th>project</th><th>label</th><th>created</th><th>expires</th><th>last used</th><th /></tr></thead>
          <tbody><For each={me()!.tokens}>{(t) => (
            <tr>
              <td class="mono">{t.project}</td><td class="lab">{t.label || '-'}</td>
              <td data-h="created"><When unix={t.created} /></td><td data-h="expires"><When unix={t.expires} never="no expiry" /></td><td data-h="last used"><When unix={t.last_used} /></td>
              <td class="act"><ConfirmButton label="revoke" confirm="revoke it" what={`token ${t.label || t.id} (${t.project})`}
                run={() => revoke.run(async () => { await api.revokeToken(t.id); await refreshMe(); showToast('token revoked'); })} /></td>
            </tr>
          )}</For></tbody>
        </table></div>
      </Show>
      <Msg err={revoke.err()} />
      <form class="prow" onSubmit={submit}>
        <label>project <select name="token-project" aria-label="token project" value={target()} onChange={(e) => setProject(e.currentTarget.value)}>
          <For each={projects()}>{(p) => <option value={p}>{p}</option>}</For></select></label>
        <input type="text" placeholder="label, e.g. cluster job" name="token-label" aria-label="token label" maxLength={60} value={label()} onInput={(e) => setLabel(e.currentTarget.value)} />
        <label>expires <select name="token-expiry" aria-label="token expiry" value={days()} onChange={(e) => setDays(e.currentTarget.value)}>
          <option value="30">in 30 days</option><option value="90">in 90 days</option><option value="365">in a year</option><option value="">never</option></select></label>
        <button type="submit" class="pri" disabled={mint.busy() || !projects().length} title="create a token for this project" aria-label="create token">{mint.busy() ? 'creating…' : 'Create token'}</button>
      </form>
      <Msg err={mint.err()} />
    </>
  );
}

function Sessions() {
  const a = useAction();
  return (
    <>
      <h2>Sessions</h2>
      <div class="ptwrap"><table class="ptable">
        <thead><tr><th>browser</th><th>signed in</th><th>last seen</th><th /></tr></thead>
        <tbody><For each={me()!.sessions}>{(s) => (
          <tr>
            <td class="lab" title={s.ua}><Show when={s.current}><span class="tag cur">this device</span> </Show>{s.ua || 'unknown'}</td>
            <td data-h="signed in"><When unix={s.created} /></td><td data-h="last seen"><When unix={s.last_seen} /></td>
            <td class="act"><ConfirmButton label={s.current ? 'sign out' : 'revoke'} confirm={s.current ? 'sign out here' : 'revoke it'} what={s.current ? 'this session' : 'session ' + s.ua.slice(0, 40)}
              run={() => a.run(async () => { if (s.current) { await signOut(); return; } await api.revokeSession(s.id); await refreshMe(); showToast('session revoked'); })} /></td>
          </tr>
        )}</For></tbody>
      </table></div>
      <Msg err={a.err()} />
    </>
  );
}

function DeleteAccount() {
  const [typed, setTyped] = createSignal('');
  const a = useAction();
  const id = () => me()!.account.id;
  const del = () => a.run(async () => { await api.deleteAccount(id()); await signOut(); showToast('account deleted'); });
  return (
    <>
      <h2>Delete account</h2>
      <div class="pdanger">
        <p class="note">Deletes the account, its sessions and tokens, and the data of every project it owns. This cannot be undone. Type the account id to confirm.</p>
        <div class="prow">
          <input type="text" class="mono" placeholder="acct_…" name="confirm-account-id" aria-label="type the account id to confirm" spellcheck={false} autocomplete="off" value={typed()} onInput={(e) => setTyped(e.currentTarget.value.trim())} />
          <button type="button" class="danger" disabled={typed() !== id() || a.busy()} title="delete the account and its projects' data" aria-label="delete this account" onClick={del}>{a.busy() ? 'deleting…' : 'Delete account'}</button>
        </div>
        <Msg err={a.err()} />
      </div>
    </>
  );
}
