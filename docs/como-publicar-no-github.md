# Atualizar o projeto no GitHub

O repositório remoto já está configurado. Antes de publicar, confirme que `.env`
e `.venv` não aparecem no status:

```powershell
git status --short
git check-ignore .env .venv
```

Depois revise, crie o commit e envie:

```powershell
git add .
git diff --cached --check
git commit -m "Organiza exemplos executáveis"
git push origin main
```

Nunca envie chaves reais. Somente arquivos `.env.example` com placeholders devem
ser versionados.
